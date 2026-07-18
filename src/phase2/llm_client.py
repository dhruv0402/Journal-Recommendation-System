import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

_client = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY not set in environment.")
        _client = Groq(api_key=api_key)
    return _client


def call_llm(prompt: str, max_tokens: int = 256) -> str:
    """
    Call Groq LLM (llama-3.1-8b-instant).
    Returns empty string on any failure — never raises.
    """
    try:
        client = _get_client()
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return ""
