import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr
from tqdm import tqdm

# === Carrega variáveis de ambiente ===
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

# Checagem da API Key
if not OPENAI_API_KEY:
    msg = "❌ OPENAI_API_KEY não encontrada. Defina no arquivo .env"
    raise ValueError(msg)

# === Caminhos ===
INPUT_FILE = Path("data/processed/constitution.jsonl")
OUTPUT_FILE = Path("data/processed/constitution_embeddings.jsonl")

if not INPUT_FILE.exists():
    msg = f"❌ Arquivo de entrada não encontrado: {INPUT_FILE}"
    raise FileNotFoundError(msg)

# === Inicializa embeddings ===
embeddings = OpenAIEmbeddings(
    model=MODEL,
    api_key=SecretStr(OPENAI_API_KEY),  # ✅ Passa sempre uma string válida
)


# === Função principal ===
def create_embeddings() -> None:
    output_data = []

    with open(INPUT_FILE, encoding="utf-8") as f:  # noqa: PTH123
        lines = f.readlines()

    print(f"📘 Gerando embeddings para {len(lines)} trechos...")

    for line in tqdm(lines, desc="Processando artigos"):
        item = json.loads(line)
        text = item["content"]

        try:
            # embed_documents retorna lista, pegamos o primeiro
            vector = embeddings.embed_documents([text])[0]
            item["embedding"] = vector
            output_data.append(item)
        except Exception as e:  # noqa: BLE001
            print(f"⚠️ Erro ao gerar embedding para {item['id']}: {e}")

    # Salva embeddings no formato JSONL
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:  # noqa: PTH123
        for item in output_data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")

    print(f"✅ Embeddings gerados e salvos em: {OUTPUT_FILE}")


# === Execução ===
if __name__ == "__main__":
    create_embeddings()
