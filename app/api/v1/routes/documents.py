from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session
from app.models.document import Document, DocumentChunkingStrategy
from app.models.embedding import EmbeddingModel
from app.services import documents
from app.db import get_db
from app.config import get_chunking_strategy, get_embedding_model


router = APIRouter()


@router.get("/{document_id}", response_model=Document)
def get_document(document_id: str, db: Session = Depends(get_db)):
    document = documents.get_document(db, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return document


@router.post("/")
def create_document(
    doc: Document,
    db: Session = Depends(get_db),
    chunking_strategy: DocumentChunkingStrategy = Depends(get_chunking_strategy),
    embedding_model: EmbeddingModel = Depends(get_embedding_model),
):
    document = documents.create_document(
        db, doc, model=embedding_model, chunking_strategy=chunking_strategy
    )
    return document


@router.delete("/")
def delete_document(doc_id: str, db: Session = Depends(get_db)):
    documents.delete_document(db, doc_id)
    return True
