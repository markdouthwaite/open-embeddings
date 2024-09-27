from app.models.embedding import Embedding, EmbeddingModel
from app.models.search import Hits, SearchResults, Chunk
from app.services.documents import get_document
from sqlmodel import Session, select, desc
from numpy import ndarray


def get_similar_embeddings(
    db: Session, query_vector: ndarray, max_k: int = 3
):
    return [
        embedding
        for embedding in db.exec(
            select(Embedding)
            .order_by(Embedding.vector.cosine_distance(query_vector))
            .limit(max_k)
        )
    ]


def get_search_results(
    db: Session, embed_model: EmbeddingModel, query_string: str, max_k: int = 3
) -> SearchResults:
    chunks = []
    refs = []
    ref_ids = set()

    for embedding in get_similar_embeddings(
        db, query_vector=embed_model(query_string), max_k=max_k
    ):
        ref_ids.add(embedding.doc_id)
        chunks.append(
            Chunk(id=embedding.id, content=embedding.chunk, doc_id=embedding.doc_id)
        )

    for doc_id in ref_ids:
        doc = get_document(db, doc_id)
        refs.append(
            {"id": doc_id, "title": doc.title, "author": doc.author, "url": doc.url}
        )

    return SearchResults(
        hits=Hits(chunks=len(chunks), refs=len(refs)), chunks=chunks, refs=refs
    )
