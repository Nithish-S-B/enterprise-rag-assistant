"""Basic grounded RAG orchestration over retrieval, reranking, and generation."""
from .citations import build_citations
from .llm import generate_text
from .reranker import rerank
from .vector_store import search


SYSTEM_INSTRUCTIONS = """You are an enterprise policy assistant.
Use only the provided retrieved context as factual evidence. Do not use outside
knowledge. Do not invent or infer unstated policy details. If the answer is not
contained in the context, clearly say that the provided documents do not contain
the answer. Do not treat examples as general policy unless the context explicitly
says they are policy. Preserve distinctions such as \"may\", \"should\", and
\"must\". Answer the user's question directly and do not mention retrieved
context, chunks, rankings, or other internal implementation details."""


def _format_context(chunks: list[dict], citations: list[dict]) -> str:
    """Format reranked chunks as distinct evidence blocks for the LLM."""
    blocks = []
    for chunk, citation in zip(chunks, citations):
        blocks.append(
            f"[CONTEXT {citation['citation_id']}]\n"
            f"Source: {citation['source']}\n"
            f"Page: {citation['page_label']}\n"
            f"Chunk ID: {citation['chunk_id']}\n\n"
            f"Content:\n{chunk['text']}"
        )
    return "\n\n".join(blocks)


def _build_prompt(question: str, chunks: list[dict], citations: list[dict]) -> str:
    """Build the complete grounded prompt for a single question."""
    return (
        "SYSTEM INSTRUCTIONS\n"
        f"{SYSTEM_INSTRUCTIONS}\n\n"
        "RETRIEVED CONTEXT\n"
        f"{_format_context(chunks, citations)}\n\n"
        "USER QUESTION\n"
        f"{question}"
    )


def answer_question(
    question: str, candidate_k: int = 10, final_k: int = 4
) -> dict:
    """Answer a question using dense retrieval, reranking, and grounded generation.

    Returns:
        A dictionary containing the generated answer and source information for
        the final reranked chunks.
    """
    if not question.strip():
        raise ValueError("question must not be empty.")
    if candidate_k < 1:
        raise ValueError("candidate_k must be at least 1.")
    if final_k < 1:
        raise ValueError("final_k must be at least 1.")

    candidates = search(question, top_k=candidate_k)
    final_chunks = rerank(question, candidates, final_k=final_k)
    citations = build_citations(final_chunks)
    prompt = _build_prompt(question, final_chunks, citations)
    answer = generate_text(prompt)

    return {"answer": answer, "citations": citations}
