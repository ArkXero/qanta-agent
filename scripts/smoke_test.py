#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.answering import QuizbowlRetriever


def main() -> None:
    retriever = QuizbowlRetriever()
    print({"docs": retriever.doc_count, "index": str(retriever.index_path)})
    tossup = retriever.predict_tossup(
        "This director used a priest's-eye view shot after a massacre in a brothel run by Sport. "
        "For 10 points, name this Italian-American director of Mean Streets and Taxi Driver."
    )
    bonus = retriever.predict_bonus(
        "Answer these questions about American Civil War battles.",
        "The winning side of this first major battle had its main position at Henry House Hill.",
    )
    print({"tossup": tossup, "bonus": bonus})
    assert "answer" in tossup and "confidence" in tossup and "buzz" in tossup
    assert "answer" in bonus and "confidence" in bonus and "explanation" in bonus


if __name__ == "__main__":
    main()
