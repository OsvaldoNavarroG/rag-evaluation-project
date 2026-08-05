import os
from functools import lru_cache

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    """
    Return a process-wide OpenAI client.

    Cached so the client is constructed at most once, on first use, rather
    than at import time. This keeps modules importable without an API key
    (e.g. for tests and tooling) and avoids constructing the client until
    it is actually needed.
    """
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to your .env file or environment."
        )
    return OpenAI(api_key=api_key)
