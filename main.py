from src.commands.publish import publish
from src.embeddings import document
from src.models.document import Document
from src.db.postgres import PostgresDocumentRepository, PostgresEmbeddingRepository

DB_STRING = "postgresql://postgres.xoqpqicateftkqglhuoi:zHIKAZOsaSB8h4YV@aws-0-eu-west-2.pooler.supabase.com:6543/postgres"

model = document.init_embedding_model("all-MiniLM-L6-v2")

with open("data/samples/blog.md") as file:
    doc = Document(content=file.read())

docs = PostgresDocumentRepository(DB_STRING)
embeds = PostgresEmbeddingRepository(DB_STRING)
