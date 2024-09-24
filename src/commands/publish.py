import uuid
from src.embeddings import document
from src.models.document import Document
from src.models.embedding import Embedding, EmbeddingModel
from src.db.base import DocumentRepository, EmbeddingsRepository


def _add_embeddings(
    model: EmbeddingModel,
    doc: Document,
    embed_repo: EmbeddingsRepository,
    width: int,
    overlap: int,
):
    for chunk, vector in document.embed(
        model, doc.content, width=width, overlap=overlap
    ):
        embedding_id = uuid.uuid4().hex
        embed_repo.add(
            Embedding(
                id=embedding_id, doc_id=doc.id, chunk=chunk, meta={}, vector=vector
            )
        )


def publish(
    model: EmbeddingModel,
    doc: Document,
    doc_repo: DocumentRepository,
    embed_repo: EmbeddingsRepository,
) -> None:
    doc_repo.add(doc)
    _add_embeddings(model, doc, embed_repo, width=1000, overlap=150)
