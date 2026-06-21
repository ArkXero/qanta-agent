#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import urllib.request
from pathlib import Path
import sys

try:
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover
    raise SystemExit("pyarrow required. Run: uv run --with pyarrow scripts/evaluate_local.py") from exc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.answering import QuizbowlRetriever, normalize_answer


def download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    req = urllib.request.Request(url, headers={"User-Agent": "qanta-agent-eval/0.1"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        target.write_bytes(resp.read())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="guesstest")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--prefix-frac", type=float, default=0.65)
    parser.add_argument("--index", type=Path, default=Path("data/qanta_index.sqlite"))
    args = parser.parse_args()

    rel = f"mode=full,char_skip=25/{args.split}-00000-of-00001.parquet"
    path = Path("data/raw/eval") / rel
    download(f"https://huggingface.co/datasets/community-datasets/qanta/resolve/main/{rel}", path)

    retriever = QuizbowlRetriever(args.index)
    total = correct = buzzes = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=512):
        for row in batch.to_pylist():
            question = row.get("full_question") or row.get("text") or ""
            if not question:
                continue
            cut = max(80, int(len(question) * args.prefix_frac))
            payload = retriever.predict_tossup(question[:cut])
            pred = normalize_answer(payload["answer"])
            golds = {normalize_answer(row.get("answer") or ""), normalize_answer(row.get("raw_answer") or "")}
            if pred and any(pred in g or g in pred for g in golds if g):
                correct += 1
            if payload["buzz"]:
                buzzes += 1
            total += 1
            if total >= args.limit:
                break
        if total >= args.limit:
            break
    print({"total": total, "accuracy": round(correct / max(total, 1), 4), "buzz_rate": round(buzzes / max(total, 1), 4)})


if __name__ == "__main__":
    main()
