import json
from pathlib import Path

from tqdm import tqdm

# Arquivo de embeddings gerados
EMBEDDINGS_FILE = Path("data/processed/constitution_embeddings.jsonl")
OUTPUT_FILE = Path("data/processed/constitution_embeddings_rastreada.jsonl")

if not EMBEDDINGS_FILE.exists():
    msg = f"❌ Arquivo de embeddings não encontrado: {EMBEDDINGS_FILE}"
    raise FileNotFoundError(msg)

updated_data = []

with open(EMBEDDINGS_FILE, encoding="utf-8") as f:  # noqa: PTH123
    lines = f.readlines()

print(f"🔍 Registrando fontes jurídicas para {len(lines)} embeddings...")

for line in tqdm(lines):
    item = json.loads(line)

    # Exemplo de extração de metadados (ajuste conforme parser que você usou)
    text = item["content"]

    # Supondo que title seja algo como "Art. 5º - Todos são iguais perante a lei"
    if "Art." in item.get("title", ""):
        article_number = item["title"].split(" ")[1].replace("º", "")
    else:
        article_number = None

    # Seção ou capítulo fictício (ajuste conforme parser)
    section = item.get("section", "Constituição Federal")

    # Número da emenda, se houver no título
    emenda = item["title"].split(" ")[1] if "Emenda" in item.get("title", "") else None

    # Adiciona metadados
    item.update(
        {
            "article_number": article_number,
            "section": section,
            "emenda": emenda,
            "jurisdiction": "Constituição Federal",
        }
    )

    updated_data.append(item)

# Salva novo arquivo com rastreabilidade completa
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:  # noqa: PTH123
    for item in updated_data:
        json.dump(item, f, ensure_ascii=False)
        f.write("\n")

print(f"✅ Metadados jurídicos registrados em: {OUTPUT_FILE}")
