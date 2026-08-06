import pytest
from rag.ingestion import ensure_nltk_resources


@pytest.fixture(scope="session", autouse=True)
def _nltk_data():
    """Ensure NLTK tokenizer data is present before any test runs."""
    ensure_nltk_resources()
