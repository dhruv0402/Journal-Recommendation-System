import os
from dotenv import load_dotenv

load_dotenv()

_client = None


def _get_client():
    global _client
    if _client is None:
        from groq import Groq
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY not set in environment.")
        _client = Groq(api_key=api_key)
    return _client


def call_llm(prompt: str, max_tokens: int = 256) -> str:
    """
    Call LLM (Groq or Ollama).
    Returns empty string on any failure — never raises.
    """
    try:
        client = _get_client()
        model = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return ""

