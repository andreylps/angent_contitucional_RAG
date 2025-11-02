import json
import os
from pathlib import Path

from dotenv import load_dotenv
from pinecone import Pinecone
from tqdm import tqdm

# === Carrega variáveis de ambiente ===
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT", "us-east1-gcp")
INDEX_NAME = os.getenv("PINECONE_INDEX", "constitution-index")

if not PINECONE_API_KEY:
    msg = "❌ PINECONE_API_KEY não definida no .env"
    raise ValueError(msg)

# Caminho do arquivo rastreado
EMBEDDINGS_FILE = Path("data/processed/constitution_embeddings_rastreada.jsonl")
if not EMBEDDINGS_FILE.exists():
    msg = f"❌ Arquivo não encontrado: {EMBEDDINGS_FILE}"
    raise FileNotFoundError(msg)

# Inicializa cliente Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Conecta ao índice existente
index = pc.Index(INDEX_NAME)


# === Função de inserção / atualização ===
def upsert_embeddings() -> None:
    vectors = []
    with open(EMBEDDINGS_FILE, encoding="utf-8") as f:  # noqa: PTH123
        lines = f.readlines()

    print(f"📥 Atualizando {len(lines)} embeddings no Pinecone...")

    batch_size = 50
    for i, line in enumerate(tqdm(lines, desc="Upsert vetores")):
        item = json.loads(line)
        vector_id = item["id"]
        vector_values = item["embedding"]
        metadata = {
            "title": item.get("title", ""),
            "source": item.get("source", ""),
            "article_number": str(item.get("article_number", "")),
            "section": item.get("section", ""),
            "emenda": str(item.get("emenda", "")),  # Substitui None por ""
            "jurisdiction": item.get("jurisdiction", ""),
        }

        vectors.append((vector_id, vector_values, metadata))

        if len(vectors) >= batch_size or i == len(lines) - 1:
            index.upsert(vectors=vectors)
            vectors = []

    print(f"✅ Todos os embeddings foram atualizados no índice: {INDEX_NAME}")


if __name__ == "__main__":
    upsert_embeddings()
