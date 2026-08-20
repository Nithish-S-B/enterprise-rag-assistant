import os
import sys

from dotenv import load_dotenv

# Ensure the backend directory is on the import path when run from the project root.
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.llm import generate_text


TEST_PROMPT = "Reply with exactly: RAG connection successful."


def test_llm_connection() -> bool:
    """Verify the configured OpenRouter model can generate a small response."""
    load_dotenv(os.path.join(backend_dir, ".env"))
    # Deliberately checks only availability and never prints the key.
    assert os.getenv("OPENROUTER_API_KEY"), "OPENROUTER_API_KEY is not configured."

    response = generate_text(TEST_PROMPT)
    print(response)
    return True


if __name__ == "__main__":
    success = test_llm_connection()
    sys.exit(0 if success else 1)
