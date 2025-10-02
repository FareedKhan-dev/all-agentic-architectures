import os
from langchain_community.vectorstores import FAISS
from .embed import make_embeddings

INDEX_DIR = os.getenv("RAG_INDEX_DIR", "./storage/faiss")


def retriever(k: int = 5):
    """Charge l'index FAISS persistant et retourne un retriever LangChain."""
    embs = make_embeddings()
    vs = FAISS.load_local(INDEX_DIR, embs, allow_dangerous_deserialization=True)
    return vs.as_retriever(search_kwargs={"k": k})
