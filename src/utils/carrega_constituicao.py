import re
from pathlib import Path


def carregar_constituicao() -> list[str]:
    """
    Lê o arquivo 'data/constitution/constituicao.md' e retorna uma lista de parágrafos limpos.
    Divide o texto com base em quebras de linha ou seções markdown.
    """

    caminho = Path("data/constitution/constituicao.md")
    if not caminho.exists():
        msg = "❌ Arquivo 'data/constitution/constituicao.md' não encontrado."
        raise FileNotFoundError(msg)

    with open(caminho, encoding="utf-8") as f:  # noqa: PTH123
        texto = f.read()

    # Remove espaços desnecessários e linhas em branco duplicadas
    texto = re.sub(r"\n{2,}", "\n\n", texto.strip())

    # Divide o texto por seções/títulos markdown ou parágrafos
    paragrafos = re.split(r"\n(?=#+ )|\n{2,}", texto)

    # Limpa e remove blocos vazios
    return [p.strip() for p in paragrafos if p.strip()]
