from typing import Any, Iterable, Tuple
from sentence_transformers import SentenceTransformer
from numpy import ndarray
from src.embeddings import chunks
from src.models.embedding import EmbeddingModel


def init_embedding_model(path: str) -> EmbeddingModel:
    model = SentenceTransformer(path)

    def _call_model(chunk: str) -> ndarray:
        embeddings = model.encode([chunk])
        return embeddings[0]

    return _call_model


def embed(
    model: EmbeddingModel, doc: str, **chunking_params: Any
) -> Iterable[Tuple[str, ndarray]]:
    for chunk in chunks.fixed_with_overlap(doc, **chunking_params):
        embedding = model(chunk)
        yield chunk, embedding
