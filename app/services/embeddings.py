from sqlmodel import Session, delete
from app.models.document import Document, DocumentChunkingStrategy
from app.models.embedding import Embedding, EmbeddingModel
from sentence_transformers import SentenceTransformer
from typing import Generator, Tuple
from numpy import ndarray


def create_embedding(db: Session, embedding: Embedding):
    db.add(embedding)
    db.commit()


def delete_embedding(db: Session, embedding_id: str):
    db.exec(delete(Embedding).where(Embedding.id == embedding_id))
    db.commit()


def get_embedding(db: Session, embedding_id: str):
    return db.get(Embedding, embedding_id)


def generate_embeddings(
    model: EmbeddingModel, chunking_strategy: DocumentChunkingStrategy, doc: Document
) -> Generator[Tuple[str, ndarray], None, None]:
    for chunk in chunking_strategy(doc.content):
        embedding = model(chunk)
        yield chunk, embedding


def init_embedding_model(path: str) -> EmbeddingModel:
    model = SentenceTransformer(path)

    def _call_model(chunk: str) -> ndarray:
        embeddings = model.encode([chunk])
        return embeddings[0]

    return _call_model
