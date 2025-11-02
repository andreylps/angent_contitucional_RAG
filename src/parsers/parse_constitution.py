import json
import re
from pathlib import Path

DATA_DIR = Path("data/constitution")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "constitution.jsonl"


def clean_markdown(text: str) -> str:
    """
    Remove formatação Markdown e caracteres desnecessários.
    """
    text = re.sub(r"#", "", text)  # remove títulos
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)  # remove negrito
    text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)  # remove links markdown
    text = re.sub(r"\n{2,}", "\n", text)  # remove múltiplas quebras de linha
    return text.strip()


def chunk_text_by_article(text: str) -> list:
    """
    Quebra o texto em blocos por artigo.
    """
    pattern = r"(Art\. ?\d+[º°]?-?.*?)(?=Art\. ?\d+[º°]?-?|$)"
    matches = re.findall(pattern, text, re.DOTALL)
    return [m.strip() for m in matches]


def parse_markdown_file(filepath: Path) -> list:
    """
    Processa um arquivo Markdown (Constituição ou Emenda).
    """
    with open(filepath, encoding="utf-8") as f:  # noqa: PTH123
        content = f.read()

    clean_text = clean_markdown(content)
    chunks = chunk_text_by_article(clean_text)

    parsed = []
    for i, chunk in enumerate(chunks, 1):
        parsed.append(
            {
                "id": f"{filepath.stem}_{i}",
                "title": chunk.split("\n")[0][:100],
                "content": chunk,
                "source": str(filepath),
            }
        )

    return parsed


def process_constitution() -> None:
    """
    Lê todos os arquivos da Constituição e gera JSONL processado.
    """
    all_data = []

    print("📘 Processando Constituição principal...")
    for file in DATA_DIR.glob("*.md"):
        all_data.extend(parse_markdown_file(file))

    print("📗 Processando emendas constitucionais...")
    for file in (DATA_DIR / "emendas").glob("*.md"):
        all_data.extend(parse_markdown_file(file))

    # Salvar resultado consolidado
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:  # noqa: PTH123
        for item in all_data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")

    print(f"✅ Processamento concluído. Arquivo salvo em: {OUTPUT_FILE}")


if __name__ == "__main__":
    process_constitution()
