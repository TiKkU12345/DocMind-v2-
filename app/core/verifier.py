import json

from app.core import generator

MAX_CLAIMS = 15


class VerifierError(Exception):
    pass


def _build_prompt(answer: str, chunks: list[dict]) -> str:
    context = "\n\n".join(
        f"[{i}] ({c['source']})\n{c['chunk']}" for i, c in enumerate(chunks, start=1)
    )
    return f"""You are a strict fact-checker. Break the ANSWER into short, atomic factual claims.
For each claim, decide whether it is directly supported by the numbered CONTEXT passages.
A claim is supported only if a passage states it or it follows directly from a passage.
Do not use outside knowledge. Ignore filler such as greetings.

Return ONLY a JSON object, no other text, in exactly this shape:
{{"claims": [{{"claim": "text of the claim", "supported": true, "chunks": [1]}}]}}
If a claim is not supported, set "supported" to false and "chunks" to [].

CONTEXT:
{context}

ANSWER:
{answer}"""


def _extract_json(raw: str) -> dict:
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        raise VerifierError("Verifier did not return JSON.")
    try:
        data = json.loads(raw[start : end + 1])
    except json.JSONDecodeError as exc:
        raise VerifierError("Verifier returned malformed JSON.") from exc
    if not isinstance(data, dict) or not isinstance(data.get("claims"), list):
        raise VerifierError("Verifier JSON is missing the claims list.")
    return data


def _normalise(data: dict, num_chunks: int) -> list[dict]:
    claims = []
    for item in data["claims"][:MAX_CLAIMS]:
        if not isinstance(item, dict) or not str(item.get("claim", "")).strip():
            continue
        supported = item.get("supported") is True
        ids = []
        if supported:
            for raw_id in item.get("chunks", []) if isinstance(item.get("chunks"), list) else []:
                if isinstance(raw_id, int) and 1 <= raw_id <= num_chunks and raw_id not in ids:
                    ids.append(raw_id)
            if not ids:
                supported = False
        claims.append({"claim": str(item["claim"]).strip(), "supported": supported, "chunks": ids})
    return claims


def verify_answer(answer: str, chunks: list[dict]) -> dict:
    raw = generator.complete(_build_prompt(answer, chunks), max_tokens=2000, temperature=0.0)
    claims = _normalise(_extract_json(raw), len(chunks))
    if not claims:
        raise VerifierError("Verifier found no claims to check.")

    supported = sum(1 for c in claims if c["supported"])
    return {
        "status": "ok",
        "claims": claims,
        "supported": supported,
        "total": len(claims),
        "faithfulness": round(supported / len(claims), 2),
    }
