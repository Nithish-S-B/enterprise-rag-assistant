"""User-facing source citation helpers for retrieved document chunks."""


def _normalize_source(source: object) -> str:
    """Return only a filename, supporting both Windows and POSIX-style paths."""
    if not isinstance(source, str) or not source:
        return "Unknown"
    return source.replace("\\", "/").rsplit("/", maxsplit=1)[-1]


def _page_label(metadata: dict) -> str:
    """Prefer loader page labels, with a one-based fallback for numeric pages."""
    label = metadata.get("page_label")
    if label is not None and str(label).strip():
        return str(label)

    page = metadata.get("page")
    if isinstance(page, int):
        return str(page + 1)
    return "Unknown"


def citation_from_result(result: dict, citation_id: str) -> dict:
    """Convert one retrieved or reranked result into a citation object."""
    metadata = result.get("metadata", {})
    return {
        "citation_id": citation_id,
        "source": _normalize_source(metadata.get("source")),
        "page": metadata.get("page"),
        "page_label": _page_label(metadata),
        "chunk_id": result.get("id"),
    }


def build_citations(results: list[dict]) -> list[dict]:
    """Build ordered, stable citation objects for final reranked results."""
    return [
        citation_from_result(result, f"S{index}")
        for index, result in enumerate(results, start=1)
    ]
