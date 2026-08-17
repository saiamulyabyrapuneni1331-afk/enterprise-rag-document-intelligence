from pathlib import Path
from typing import List

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document


def load_document(file_path: str) -> List[Document]:
    """
    Load a PDF or TXT document and return LangChain Documents.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")

    extension = path.suffix.lower()

    if extension == ".pdf":
        loader = PyPDFLoader(str(path))

    elif extension == ".txt":
        loader = TextLoader(
            str(path),
            encoding="utf-8",
        )

    else:
        raise ValueError(
            f"Unsupported file type: {extension}. "
            "Only PDF and TXT files are supported."
        )

    return loader.load()


def load_directory(directory_path: str) -> List[Document]:
    """
    Load all supported documents from a directory.
    """

    directory = Path(directory_path)

    if not directory.exists():
        raise FileNotFoundError(
            f"Directory not found: {directory_path}"
        )

    documents: List[Document] = []

    for file_path in directory.iterdir():

        if file_path.suffix.lower() in {".pdf", ".txt"}:
            documents.extend(
                load_document(str(file_path))
            )

    return documents
