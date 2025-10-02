import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings


def make_embeddings(provider: str | None = None, model: str | None = None):
    """Fabrique l'encodeur d'embeddings (Gemini par défaut)."""
    p = (provider or os.getenv("DEFAULT_EMBEDDING_PROVIDER", "gemini")).lower()
    m = model or os.getenv("DEFAULT_EMBEDDING_MODEL", "text-embedding-004")
    if p == "openai":
        return OpenAIEmbeddings(model=m)  # ex: text-embedding-3-large
    return GoogleGenerativeAIEmbeddings(model=m)
