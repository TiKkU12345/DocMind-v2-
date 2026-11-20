from groq import Groq

from app.config import settings

_client = None


def _get_client() -> Groq:
    global _client
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not set on the server.")
    if _client is None:
        _client = Groq(api_key=settings.groq_api_key)
    return _client


def generate_answer(query: str, chunks: list[dict]) -> str:
    context = "\n\n".join(f"[{c['source']}]\n{c['chunk']}" for c in chunks)

    prompt = f"""You are a helpful AI assistant. Answer the user's question based only on the provided context.
If the answer is not in the context, say "I don't have enough information to answer this."

Context:
{context}

Question: {query}

Answer:"""

    extra = {}
    if settings.groq_model.startswith("openai/gpt-oss"):
        extra["extra_body"] = {"reasoning_effort": "low"}

    response = _get_client().chat.completions.create(
        model=settings.groq_model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000,
        temperature=0.1,
        **extra,
    )
    answer = (response.choices[0].message.content or "").strip()
    if not answer:
        raise RuntimeError("The model returned an empty answer.")
    return answer
