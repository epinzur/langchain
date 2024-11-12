METADATA_EMBEDDING_KEY = "__embedding"

from langchain_core.documents import Document


def get_embedding(doc: Document) -> list[float]:
    """Get the embedding from a document."""
    return doc.metadata.get(METADATA_EMBEDDING_KEY, [])


def set_embedding(doc: Document, embedding: list[float]) -> None:
    """Set the embedding on a document."""
    doc.metadata[METADATA_EMBEDDING_KEY] = embedding


def clear_embedding(doc: Document) -> None:
    doc.metadata.pop(METADATA_EMBEDDING_KEY, None)
