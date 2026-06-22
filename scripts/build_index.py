#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sqlite3
import tempfile
import urllib.request
from pathlib import Path
from typing import Iterable

try:
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover
    raise SystemExit("pyarrow required. Run: uv run --with pyarrow scripts/build_index.py") from exc


QANTA_SPLITS = (
    "guesstrain",
    "guesstest",
    "buzztrain",
    "buzztest",
    "guessdev",
    "buzzdev",
    "adversarial",
)

QANTA_FILES = [
    ("community-datasets/qanta", f"mode={mode},char_skip=25/{split}-00000-of-00001.parquet")
    for mode in ("full", "first")
    for split in QANTA_SPLITS
]
QANTA_FILES.append(("mgor/protobowl-11-13", "questions/eval-00000-of-00001.parquet"))


def hf_resolve_url(repo: str, path: str) -> str:
    return f"https://huggingface.co/datasets/{repo}/resolve/main/{path}"


def download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        return
    tmp = target.with_suffix(target.suffix + ".tmp")
    print(f"download {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "qanta-agent-builder/0.1"})
    with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("wb") as out:
        out.write(resp.read())
    tmp.replace(target)


def row_docs(path: Path) -> Iterable[dict]:
    pf = pq.ParquetFile(path)
    prefer_text = "mode=first" in str(path)
    for batch in pf.iter_batches(batch_size=2048):
        table = batch.to_pylist()
        for row in table:
            if "full_question" in row:
                if prefer_text:
                    question = row.get("text") or row.get("first_sentence") or row.get("full_question") or ""
                else:
                    question = row.get("full_question") or row.get("text") or ""
                answer = row.get("answer") or row.get("page") or ""
                raw = row.get("raw_answer") or answer
                qid = str(row.get("qanta_id") or row.get("id") or "")
                first = row.get("first_sentence") or ""
                category = row.get("category") or ""
                source = f"{row.get('fold') or path.stem}:{path.parent.name}"
            else:
                question = row.get("question") or ""
                answer = row.get("answer_primary") or row.get("wiki_page") or row.get("answer") or ""
                raw = row.get("answer") or answer
                qid = str(row.get("qid") or "")
                first = question[:220]
                category = row.get("category") or ""
                source = row.get("question_set") or path.stem
            if not question or not answer:
                continue
            yield {
                "source_id": qid,
                "answer": str(answer),
                "raw_answer": str(raw),
                "full_question": str(question),
                "first_sentence": str(first),
                "category": str(category),
                "source": str(source),
            }


def build_index(raw_dir: Path, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix="qanta_index_", suffix=".sqlite", dir=str(out_path.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    con = sqlite3.connect(tmp_path)
    con.execute("PRAGMA journal_mode=OFF")
    con.execute("PRAGMA synchronous=OFF")
    con.execute(
        """
        CREATE TABLE docs (
            id INTEGER PRIMARY KEY,
            source_id TEXT,
            answer TEXT NOT NULL,
            raw_answer TEXT,
            full_question TEXT NOT NULL,
            first_sentence TEXT,
            category TEXT,
            source TEXT
        )
        """
    )
    con.execute(
        """
        CREATE VIRTUAL TABLE docs_fts USING fts5(
            full_question,
            first_sentence,
            answer,
            raw_answer,
            category,
            content='docs',
            content_rowid='id',
            tokenize='unicode61 remove_diacritics 2'
        )
        """
    )
    con.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")

    seen: set[tuple[str, str]] = set()
    inserted = 0
    for repo, rel in QANTA_FILES:
        local = raw_dir / repo.replace("/", "__") / rel
        download(hf_resolve_url(repo, rel), local)
        for doc in row_docs(local):
            key = (doc["answer"], doc["full_question"][:500])
            if key in seen:
                continue
            seen.add(key)
            cur = con.execute(
                """
                INSERT INTO docs (source_id, answer, raw_answer, full_question, first_sentence, category, source)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    doc["source_id"],
                    doc["answer"],
                    doc["raw_answer"],
                    doc["full_question"],
                    doc["first_sentence"],
                    doc["category"],
                    doc["source"],
                ),
            )
            rowid = cur.lastrowid
            con.execute(
                """
                INSERT INTO docs_fts (rowid, full_question, first_sentence, answer, raw_answer, category)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    rowid,
                    doc["full_question"],
                    doc["first_sentence"],
                    doc["answer"],
                    doc["raw_answer"],
                    doc["category"],
                ),
            )
            inserted += 1
            if inserted % 5000 == 0:
                print(f"indexed {inserted}")
                con.commit()

    con.execute("INSERT INTO meta (key, value) VALUES ('docs', ?)", (str(inserted),))
    con.execute("INSERT INTO meta (key, value) VALUES ('sources', ?)", (",".join(f"{r}/{p}" for r, p in QANTA_FILES),))
    con.commit()
    con.execute("VACUUM")
    con.close()
    tmp_path.replace(out_path)
    print(f"wrote {out_path} with {inserted} docs")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/qanta_index.sqlite"))
    args = parser.parse_args()
    build_index(args.raw_dir, args.out)


if __name__ == "__main__":
    main()
