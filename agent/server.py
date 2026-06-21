from __future__ import annotations

from functools import lru_cache
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .answering import QuizbowlRetriever


class TossupRequest(BaseModel):
    question_text: str = ""
    images: list[Any] = Field(default_factory=list)
    question_images: list[Any] = Field(default_factory=list)


class BonusRequest(BaseModel):
    leadin: str = ""
    part: str = ""
    images: list[Any] = Field(default_factory=list)
    leadin_images: list[Any] = Field(default_factory=list)
    part_images: list[Any] = Field(default_factory=list)


@lru_cache(maxsize=1)
def get_retriever() -> QuizbowlRetriever:
    return QuizbowlRetriever()


app = FastAPI(title="QANTA Docker Agent", version="0.1.0")


@app.get("/health")
def health() -> dict[str, Any]:
    retriever = get_retriever()
    return {"status": "ok", "docs": retriever.doc_count, "index": str(retriever.index_path)}


@app.post("/predict/tossup")
def predict_tossup(payload: TossupRequest) -> dict[str, Any]:
    images = payload.images or payload.question_images
    return get_retriever().predict_tossup(payload.question_text, images)


@app.post("/predict/bonus")
def predict_bonus(payload: BonusRequest) -> dict[str, Any]:
    images = payload.images or payload.leadin_images + payload.part_images
    return get_retriever().predict_bonus(payload.leadin, payload.part, images)
