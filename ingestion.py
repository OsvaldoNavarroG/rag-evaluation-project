from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

def load_documents(path):
    with open(path, "r") as f:
        return f.read()
    
def chunk_text(text, chunk_size=200)->List[str]:
    words=text.split()
    chunks=[]
    for i in range(0, len(words), chunk_size):
        chunks.append("".join(words[i:i + chunk_size]))
    return chunks

def embed_chunks(chunks):
    embeddings = model.encode(chunks)
    return np.array(embeddings)

def build_index(embeddings):
    import faiss
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index