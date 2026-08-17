from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.ingestion.embeddings import get_embedding_model


VECTOR_STORE_PATH = "vector_store"


def create_vector_store(documents: list[Document]) -> FAISS:
    """
    Create a FAISS vector database from document chunks.
    """

    if not documents:
        raise ValueError("No documents were provided.")

    embeddings = get_embedding_model()

    vector_store = FAISS.from_documents(
        documents=documents,
        embedding=embeddings,
    )

    return vector_store


def save_vector_store(
    vector_store: FAISS,
    path: str = VECTOR_STORE_PATH,
) -> None:
    """
    Save the FAISS vector database locally.
    """

    Path(path).mkdir(
        parents=True,
        exist_ok=True,
    )

    vector_store.save_local(path)


def load_vector_store(
    path: str = VECTOR_STORE_PATH,
) -> FAISS:
    """
    Load an existing FAISS vector database.
    """

    embeddings = get_embedding_model()

    return FAISS.load_local(
        path,
        embeddings,
        allow_dangerous_deserialization=True,
    )
