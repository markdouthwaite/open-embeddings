import uuid
from typing import Callable, Any, Dict
from numpy import ndarray
from sqlmodel import Field, SQLModel
from sqlalchemy import Column, JSON
from datetime import datetime, timezone
from pgvector.sqlalchemy import Vector

EmbeddingModel = Callable[[str], ndarray]


class Embedding(SQLModel, table=True):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex, primary_key=True)
    doc_id: str
    chunk: str
    vector: Any = Field(sa_column=Column(Vector(384)))
    meta: Dict = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
