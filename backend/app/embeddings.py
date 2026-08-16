"""
Local embedding generation using Sentence Transformers.

The embedding model converts text chunks and queries into fixed-size
vectors so semantically similar text can be compared by distance.
"""
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Load the model once at import time and reuse it for every embedding call.
_model = SentenceTransformer(MODEL_NAME)


def embed_text(text: str) -> list[float]:
    """
    Generates a 384-dimensional embedding vector for the supplied text.

    Args:
        text (str): The text to embed (e.g., a document chunk or a user query).

    Returns:
        list[float]: The embedding vector for the supplied text.
    """
    return _model.encode(text).tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Generates one embedding per input text in a single batched call.

    Args:
        texts (list[str]): The texts to embed (e.g., document chunks).

    Returns:
        list[list[float]]: One 384-dimensional embedding per input text,
            in the same order as the inputs.
    """
    return _model.encode(texts).tolist()
