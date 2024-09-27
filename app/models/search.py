from typing import List
from pydantic import BaseModel

from .document import DocumentReference


class Chunk(BaseModel):
    id: str
    content: str
    doc_id: str


class Hits(BaseModel):
    chunks: int
    refs: int


class SearchResults(BaseModel):
    hits: Hits
    chunks: List[Chunk]
    refs: List[DocumentReference]
