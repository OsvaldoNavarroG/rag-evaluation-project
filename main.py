from typing import List
from rag.ingestion import load_documents
from rag.evaluation import compare_chunking_approaches


# 1. Load and prepare data
text: str = load_documents(path="data/docs.txt")

# 2. Test queries (you will expand this later)
test_data: List[dict] = [
    {
        "question": "What is the main topic of the document?",
        "expected": "machine learning",
    }
]

compare_chunking_approaches(text=text, test_data=test_data)
