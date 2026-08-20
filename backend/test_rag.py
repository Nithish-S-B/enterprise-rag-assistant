import sys

from app.rag import answer_question


QUESTIONS = [
    "Who approves leave requests?",
    "How many annual leave days do employees get?",
    "How many days can employees work remotely?",
]


def test_rag() -> bool:
    """Run basic grounded-RAG checks against representative policy questions."""
    for question in QUESTIONS:
        result = answer_question(question, candidate_k=10, final_k=4)
        answer = result["answer"]
        sources = result["sources"]

        assert answer.strip(), "Expected a non-empty generated answer."
        assert sources, "Expected retrieved source information."

        print("\n" + "=" * 80)
        print(f"QUESTION: {question}")
        print("RETRIEVED SOURCES:")
        for source in sources:
            print(
                f"- source={source['source']} | page={source['page']} | "
                f"chunk_id={source['chunk_id']}"
            )
        print("GENERATED ANSWER:")
        print(answer)

    print("\n" + "=" * 80)
    print("RAG test complete.")
    return True


if __name__ == "__main__":
    success = test_rag()
    sys.exit(0 if success else 1)
