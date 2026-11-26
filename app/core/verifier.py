import json

from app.core import generator

MAX_CLAIMS = 15


class VerifierError(Exception):
    pass


def _build_prompt(answer: str, chunks: list[dict]) -> str:
    context = "\n\n".join(
        f"[{i}] ({c['source']})\n{c['chunk']}" for i, c in enumerate(chunks, start=1)
    )
    return f"""You are a strict fact-checker. Break the ANSWER into short, atomic claims.
Ignore filler such as greetings. Give every claim a "kind":

1. kind "fact": a positive statement. It is supported only if a numbered CONTEXT passage states it or it follows directly from a passage. Do not use outside knowledge. List the supporting passage numbers in "chunks". If no passage supports it, set "supported" to false and "chunks" to [].

2. kind "absence": a statement that something is NOT mentioned, NOT covered or NOT stated (for example "the agreement does not mention a discount"). Set "supported" to true if none of the CONTEXT passages mention that thing. If a passage does mention it, set "supported" to false and list that passage in "chunks".

Return ONLY a JSON object, no other text, in exactly this shape:
{{"claims": [{{"claim": "text of the claim", "kind": "fact", "supported": true, "chunks": [1]}}]}}

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


def _valid_ids(raw, num_chunks: int) -> list[int]:
    ids = []
    if isinstance(raw, list):
        for value in raw:
            if isinstance(value, int) and not isinstance(value, bool) and 1 <= value <= num_chunks and value not in ids:
                ids.append(value)
    return ids


def _normalise(data: dict, num_chunks: int) -> list[dict]:
    claims = []
    for item in data["claims"][:MAX_CLAIMS]:
        if not isinstance(item, dict) or not str(item.get("claim", "")).strip():
            continue
        kind = "absence" if item.get("kind") == "absence" else "fact"
        supported = item.get("supported") is True
        ids = _valid_ids(item.get("chunks"), num_chunks)

        if kind == "fact":
            if supported and not ids:
                supported = False
            if not supported:
                ids = []
        elif supported:
            ids = []

        claims.append(
            {"claim": str(item["claim"]).strip(), "kind": kind, "supported": supported, "chunks": ids}
        )
    return claims


def verify_answer(answer: str, chunks: list[dict]) -> dict:
    raw = generator.complete(_build_prompt(answer, chunks), max_tokens=2000, temperature=0.0)
    claims = _normalise(_extract_json(raw), len(chunks))
    if not claims:
        raise VerifierError("Verifier found no claims to check.")

    facts = [c for c in claims if c["kind"] == "fact"]
    absences = [c for c in claims if c["kind"] == "absence"]
    supported = sum(1 for c in facts if c["supported"])

    return {
        "status": "ok",
        "claims": claims,
        "supported": supported,
        "total": len(facts),
        "faithfulness": round(supported / len(facts), 2) if facts else None,
        "absence_claims": len(absences),
        "absence_contradicted": sum(1 for c in absences if not c["supported"]),
    }
