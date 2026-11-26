import json

import pytest

from app.core import generator, verifier

CHUNKS = [
    {"chunk": "Payment is due within 30 days.", "source": "a.txt", "score": 0.5},
    {"chunk": "Late interest is 1.5 percent per month.", "source": "a.txt", "score": 0.4},
]


def fake_llm(payload):
    def _complete(prompt, max_tokens=2000, temperature=0.1):
        return payload if isinstance(payload, str) else json.dumps(payload)

    return _complete


def test_faithfulness_is_computed(monkeypatch):
    payload = {
        "claims": [
            {"claim": "Payment is due in 30 days", "supported": True, "chunks": [1]},
            {"claim": "Interest is 1.5 percent monthly", "supported": True, "chunks": [2]},
            {"claim": "There is a 10 percent discount", "supported": False, "chunks": []},
        ]
    }
    monkeypatch.setattr(generator, "complete", fake_llm(payload))
    result = verifier.verify_answer("whatever", CHUNKS)
    assert result["supported"] == 2
    assert result["total"] == 3
    assert result["faithfulness"] == 0.67


def test_handles_code_fences_and_chatter(monkeypatch):
    payload = '```json\n{"claims": [{"claim": "x", "supported": true, "chunks": [1]}]}\n```\nHope that helps'
    monkeypatch.setattr(generator, "complete", fake_llm(payload))
    assert verifier.verify_answer("a", CHUNKS)["faithfulness"] == 1.0


def test_supported_claim_with_invalid_chunk_ids_is_downgraded(monkeypatch):
    payload = {"claims": [{"claim": "x", "supported": True, "chunks": [9, 0, "1"]}]}
    monkeypatch.setattr(generator, "complete", fake_llm(payload))
    result = verifier.verify_answer("a", CHUNKS)
    assert result["claims"][0]["supported"] is False
    assert result["faithfulness"] == 0.0


def test_only_boolean_true_counts_as_supported(monkeypatch):
    payload = {"claims": [{"claim": "x", "supported": "yes", "chunks": [1]}]}
    monkeypatch.setattr(generator, "complete", fake_llm(payload))
    assert verifier.verify_answer("a", CHUNKS)["claims"][0]["supported"] is False


@pytest.mark.parametrize("bad", ["not json at all", "{broken", '{"claims": "nope"}', '{"claims": []}'])
def test_bad_output_raises(monkeypatch, bad):
    monkeypatch.setattr(generator, "complete", fake_llm(bad))
    with pytest.raises(verifier.VerifierError):
        verifier.verify_answer("a", CHUNKS)


def test_abstention_detection():
    assert generator.is_abstention("I don't have enough information to answer this.")
    assert generator.is_abstention("I don\u2019t have enough information to answer this.")
    assert not generator.is_abstention("Payment is due in 30 days.")
