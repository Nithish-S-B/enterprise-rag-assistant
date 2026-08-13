from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def chunk_documents(documents: list[Document]) -> list[Document]:
    """
    Splits page-level documents into retrieval-ready chunks.

    Uses RecursiveCharacterTextSplitter to break documents at natural
    boundaries (paragraphs, newlines, spaces) while preserving each
    document's metadata on every resulting chunk.

    Args:
        documents (list[Document]): A list of LangChain Document objects
            (e.g., one per PDF page).

    Returns:
        list[Document]: A list of LangChain Document objects, each being
            a chunk with the parent document's metadata intact.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(documents)
