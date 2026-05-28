"""Vector-store abstraction (FAISS / Chroma / Qdrant).

The RAG notebooks (Agentic / Corrective / Self / Adaptive / GraphRAG) all
use this — switching backends is a one-line `.env` change.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agentic_architectures.config import VectorBackend, settings

if TYPE_CHECKING:
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
    from langchain_core.vectorstores import VectorStore


def get_vector_store(
    documents: list[Document] | None = None,
    embeddings: Embeddings | None = None,
    backend: VectorBackend | None = None,
    collection_name: str = "agentic_architectures",
    **kwargs: Any,
) -> VectorStore:
    """Return a vector store of the configured backend, optionally pre-populated."""
    backend = backend or settings.vector_backend

    if embeddings is None:
        from agentic_architectures.llm.factory import get_embeddings

        embeddings = get_embeddings()

    if backend == "faiss":
        try:
            from langchain_community.vectorstores import FAISS
        except ImportError as e:
            raise ImportError("pip install agentic-architectures[faiss]") from e
        if documents:
            return FAISS.from_documents(documents, embeddings, **kwargs)
        # Empty FAISS store: add a sentinel doc then delete it (FAISS requires
        # at least one doc to instantiate). Caller is expected to call .add_documents().
        from langchain_core.documents import Document as _Doc

        store = FAISS.from_documents([_Doc(page_content="__sentinel__")], embeddings)
        store.delete([store.index_to_docstore_id[0]])
        return store

    if backend == "chroma":
        try:
            from langchain_chroma import Chroma
        except ImportError as e:
            raise ImportError("pip install agentic-architectures[chroma]") from e
        store = Chroma(
            collection_name=collection_name, embedding_function=embeddings, **kwargs
        )
        if documents:
            store.add_documents(documents)
        return store

    if backend == "qdrant":
        try:
            from langchain_qdrant import QdrantVectorStore
        except ImportError as e:
            raise ImportError("pip install agentic-architectures[qdrant]") from e
        api_key = (
            settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
        )
        if documents:
            return QdrantVectorStore.from_documents(
                documents,
                embedding=embeddings,
                url=settings.qdrant_url,
                api_key=api_key,
                collection_name=collection_name,
                **kwargs,
            )
        return QdrantVectorStore.from_existing_collection(
            embedding=embeddings,
            url=settings.qdrant_url,
            api_key=api_key,
            collection_name=collection_name,
            **kwargs,
        )

    raise ValueError(f"Unknown vector backend: {backend!r}")


class VectorMemory:
    """Thin convenience wrapper used by RAG architectures."""

    def __init__(
        self,
        embeddings: Embeddings | None = None,
        backend: VectorBackend | None = None,
        collection_name: str = "agentic_architectures",
    ) -> None:
        self._store = get_vector_store(
            embeddings=embeddings,
            backend=backend,
            collection_name=collection_name,
        )

    @property
    def store(self) -> VectorStore:
        return self._store

    def add(self, documents: list[Document]) -> list[str]:
        return self._store.add_documents(documents)

    def search(self, query: str, k: int = 4) -> list[Document]:
        return self._store.similarity_search(query, k=k)

    def as_retriever(self, **kwargs: Any) -> Any:
        return self._store.as_retriever(**kwargs)
