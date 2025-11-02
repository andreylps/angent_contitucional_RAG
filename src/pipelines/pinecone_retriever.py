import json
import os
from pathlib import Path

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm

# === Carrega variáveis de ambiente ===
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT", "us-east1-gcp")
INDEX_NAME = os.getenv("PINECONE_INDEX", "constitution-index")

if not PINECONE_API_KEY:
    msg = "❌ PINECONE_API_KEY não definida no .env"
    raise ValueError(msg)

# Caminho dos embeddings
EMBEDDINGS_FILE = Path("data/processed/constitution_embeddings.jsonl")
if not EMBEDDINGS_FILE.exists():
    msg = f"❌ Arquivo de embeddings não encontrado: {EMBEDDINGS_FILE}"
    raise FileNotFoundError(msg)

# === Inicializa cliente Pinecone ===
pc = Pinecone(api_key=PINECONE_API_KEY)

# === Cria índice caso não exista ===
existing_indexes = [idx.name for idx in pc.list_indexes()]
if INDEX_NAME not in existing_indexes:
    print(f"📌 Criando índice Pinecone: {INDEX_NAME}")
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,  # text-embedding-3-small
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws", region="us-east-1"
        ),  # Ajuste região se necessário
    )
else:
    print(f"📌 Índice existente: {INDEX_NAME}")

# Conecta ao índice
index = pc.Index(INDEX_NAME)


# === Função de inserção ===
def insert_embeddings() -> None:
    vectors = []
    with open(EMBEDDINGS_FILE, encoding="utf-8") as f:  # noqa: PTH123
        lines = f.readlines()

    print(f"📥 Inserindo {len(lines)} embeddings no Pinecone...")

    batch_size = 50
    for i, line in enumerate(tqdm(lines, desc="Inserindo vetores")):
        item = json.loads(line)
        vector_id = item["id"]
        vector_values = item["embedding"]
        metadata = {"title": item.get("title"), "source": item.get("source")}
        vectors.append((vector_id, vector_values, metadata))

        if len(vectors) >= batch_size or i == len(lines) - 1:
            index.upsert(vectors=vectors)
            vectors = []

    print(f"✅ Todos os embeddings foram inseridos no índice: {INDEX_NAME}")


if __name__ == "__main__":
    insert_embeddings()
