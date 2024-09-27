import uuid
from sqlmodel import Field, SQLModel
from typing import Optional, Dict, Callable, Iterable
from sqlalchemy import Column, JSON
from datetime import datetime, timezone


class Document(SQLModel, table=True):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex, primary_key=True)
    title: Optional[str]
    url: Optional[str]
    content: str
    author: Optional[str]
    meta: Dict = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DocumentReference(SQLModel):
    id: str
    url: Optional[str]
    author: Optional[str]
    title: Optional[str]
    meta: Dict = Field(default_factory=dict, sa_column=Column(JSON))


DocumentChunkingStrategy = Callable[[str], Iterable[str]]
