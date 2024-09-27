from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session
from app.models.search import Chunk
from app.services import embeddings
from app.db import get_db


router = APIRouter()


@router.get("/{embedding_id}", response_model=Chunk)
def get_embedding(embedding_id: str, db: Session = Depends(get_db)):
    embedding = embeddings.get_embedding(db, embedding_id)
    if not embedding:
        raise HTTPException(status_code=404, detail="Embedding not found")
    return Chunk(id=embedding.id, content=embedding.chunk, doc_id=embedding.doc_id)


@router.delete("/")
def delete_embedding(embedding_id: str, db: Session = Depends(get_db)):
    embeddings.delete_embedding(db, embedding_id)
    return True
