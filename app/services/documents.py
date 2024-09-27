from sqlmodel import Session, select, delete
from typing import Dict, Tuple
from app.models.document import Document, DocumentChunkingStrategy
from app.models.embedding import Embedding, EmbeddingModel
from app.services.embeddings import (
    create_embedding,
    generate_embeddings,
    delete_embedding,
)


def create_document(
    db: Session,
    doc: Document,
    model: EmbeddingModel,
    chunking_strategy: DocumentChunkingStrategy,
) -> Tuple[str, int, int]:
    db.add(doc)
    db.commit()
    for i, (chunk, vector) in enumerate(
        generate_embeddings(model, chunking_strategy, doc)
    ):
        create_embedding(
            db, Embedding(doc_id=doc.id, chunk=chunk, meta=doc.meta, vector=vector)
        )

    return (doc.id, 1, i + 1)


def get_document(db: Session, doc_id: str) -> Document:
    return db.get(Document, doc_id)


def _delete_associated_embeddings(db: Session, doc_id: str) -> int:
    if db.get(Document, doc_id) is not None:
        for i, embedding in enumerate(
            db.exec(select(Embedding).where(Embedding.doc_id == doc_id))
        ):
            delete_embedding(db, embedding.id)
        return i + 1
    return 0


def delete_document(db: Session, doc_id: str) -> Tuple[int, int]:
    n_deleted_embeddings = _delete_associated_embeddings(db, doc_id)
    db.exec(delete(Document).where(Document.id == doc_id))
    db.commit()
    return 1, n_deleted_embeddings
