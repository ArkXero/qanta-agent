from __future__ import annotations

import math
import os
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_env_index = os.environ.get("QANTA_INDEX_PATH")
DEFAULT_INDEX_PATHS = tuple(
    path
    for path in (
        Path(_env_index) if _env_index else None,
        Path("/app/data/qanta_index.sqlite"),
        Path(__file__).resolve().parents[1] / "data" / "qanta_index.sqlite",
    )
    if path is not None
)

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "its",
    "name",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "what",
    "which",
    "who",
    "whose",
    "with",
}


def _first_existing(paths: tuple[Path, ...]) -> Path:
    for path in paths:
        if path and path.exists():
            return path
    searched = ", ".join(str(p) for p in paths if p)
    raise FileNotFoundError(f"QANTA index not found. Searched: {searched}")


def clean_text(text: str) -> str:
    text = text or ""
    text = re.sub(r"<multimodal\b[^>]*>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\([^)]{1,80}\)", " ", text)
    text = re.sub(r"\[[^\]]{1,120}\]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def display_answer(answer: str, raw_answer: str | None = None) -> str:
    candidate = raw_answer or answer or ""
    candidate = re.split(r"\s+\[|\s+\(|\s+prompt\s+on|\s+accept\s+", candidate, maxsplit=1, flags=re.IGNORECASE)[0]
    candidate = candidate.replace("_", " ")
    candidate = candidate.replace("{", "").replace("}", "")
    candidate = re.sub(r"\s*/\s*", " / ", candidate)
    candidate = re.sub(r"\s+", " ", candidate).strip(" ;,")
    return candidate or (answer or "unknown").replace("_", " ")


def normalize_answer(answer: str) -> str:
    answer = display_answer(answer)
    answer = answer.lower()
    answer = re.sub(r"[^a-z0-9 ]+", " ", answer)
    answer = re.sub(r"\b(the|a|an)\b", " ", answer)
    return re.sub(r"\s+", " ", answer).strip()


def query_terms(text: str, limit: int = 20) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9_'-]{2,}", clean_text(text).lower())
    terms: list[str] = []
    seen: set[str] = set()
    for word in reversed(words):
        word = word.strip("_'-")
        if len(word) < 3 or word in STOPWORDS or word in seen:
            continue
        seen.add(word)
        terms.append(word)
        if len(terms) >= limit:
            break
    return list(reversed(terms))


def fts_queries(text: str) -> list[str]:
    terms = query_terms(text, limit=14)
    if not terms:
        return ['""']
    strong = [term for term in terms if len(term) >= 5][-8:] or terms[-8:]
    broad = terms[-10:]

    def quote_terms(items: list[str], joiner: str) -> str:
        return joiner.join(f'"{term.replace('"', "")}"' for term in items)

    queries = [quote_terms(strong, " ")]
    if broad != strong:
        queries.append(quote_terms(broad, " OR "))
    return queries


@dataclass
class RetrievalHit:
    answer: str
    raw_answer: str
    question: str
    category: str
    score: float
    confidence: float


class QuizbowlRetriever:
    def __init__(self, index_path: Path | None = None) -> None:
        self.index_path = index_path or _first_existing(DEFAULT_INDEX_PATHS)
        self.conn = sqlite3.connect(str(self.index_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._doc_count = self.conn.execute("SELECT COUNT(*) FROM docs").fetchone()[0]

    @property
    def doc_count(self) -> int:
        return int(self._doc_count)

    def search(self, text: str, limit: int = 5) -> list[RetrievalHit]:
        cleaned = clean_text(text)
        if not query_terms(cleaned):
            return []
        rows = []
        for q in fts_queries(cleaned):
            rows = self.conn.execute(
                """
                SELECT
                    docs.answer,
                    docs.raw_answer,
                    docs.full_question,
                    docs.category,
                    bm25(docs_fts, 9.0, 3.0, 1.5, 0.4) AS rank
                FROM docs_fts
                JOIN docs ON docs_fts.rowid = docs.id
                WHERE docs_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (q, limit),
            ).fetchall()
            if rows:
                break
        if not rows:
            return []

        ranks = [float(row["rank"]) for row in rows]
        best = -ranks[0]
        second = -ranks[1] if len(ranks) > 1 else best - 1.5
        gap = max(0.0, best - second)
        n_tokens = len(re.findall(r"\w+", cleaned))
        late_bonus = 0.14 if re.search(r"\bfor\s+10\s+points\b|\bname\s+this\b|\bidentify\s+this\b", cleaned, re.I) else 0.0
        length_bonus = min(0.18, max(0.0, (n_tokens - 18) / 160))
        base_conf = 0.24 + 0.35 * math.tanh(best / 7.5) + 0.22 * math.tanh(gap / 2.5)
        confidence = max(0.04, min(0.97, base_conf + length_bonus + late_bonus))

        hits: list[RetrievalHit] = []
        for i, row in enumerate(rows):
            row_score = -float(row["rank"])
            row_conf = confidence if i == 0 else max(0.03, confidence - 0.12 - 0.04 * i)
            hits.append(
                RetrievalHit(
                    answer=display_answer(row["answer"], row["raw_answer"]),
                    raw_answer=row["raw_answer"] or "",
                    question=row["full_question"] or "",
                    category=row["category"] or "",
                    score=row_score,
                    confidence=row_conf,
                )
            )
        return hits

    def predict_tossup(self, question_text: str, images: list[Any] | None = None) -> dict[str, Any]:
        hits = self.search(question_text)
        if not hits:
            return {"answer": "", "confidence": 0.02, "buzz": False}
        best = hits[0]
        threshold = float(os.environ.get("TOSSUP_BUZZ_THRESHOLD", "0.62"))
        if images:
            threshold += float(os.environ.get("IMAGE_ONLY_CAUTION", "0.03"))
        return {
            "answer": best.answer,
            "confidence": round(best.confidence, 4),
            "buzz": best.confidence >= threshold,
        }

    def predict_bonus(self, leadin: str, part: str, images: list[Any] | None = None) -> dict[str, Any]:
        query = f"{leadin} {part}".strip()
        hits = self.search(query)
        if not hits:
            return {
                "answer": "",
                "confidence": 0.02,
                "explanation": "No close match found in the local quizbowl index.",
            }
        best = hits[0]
        confidence = max(0.05, min(0.95, best.confidence - 0.04))
        if images:
            confidence = max(0.05, confidence - float(os.environ.get("IMAGE_ONLY_CAUTION", "0.03")))
        clue = clean_text(best.question)[:260]
        explanation = (
            f"Local retrieval matched a {best.category or 'quizbowl'} clue pointing to {best.answer}. "
            f"Closest indexed clue: {clue}"
        )
        return {
            "answer": best.answer,
            "confidence": round(confidence, 4),
            "explanation": explanation,
        }
