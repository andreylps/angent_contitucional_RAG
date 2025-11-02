# src/pipelines/rag_bm25.py

import os
import re

from rank_bm25 import BM25Okapi


# ------------------------------
# CARREGAR E LIMPAR CONSTITUIÇÃO
# ------------------------------
def load_constitution():  # noqa: ANN201
    """Carrega o arquivo Markdown e retorna o texto limpo"""
    base_path = os.path.join(  # noqa: PTH118
        os.path.dirname(__file__),  # noqa: PTH120
        "..",
        "..",
        "data",
        "constitution",
        "constituicao.md",
    )

    if not os.path.exists(base_path):  # noqa: PTH110
        msg = f"Arquivo não encontrado: {base_path}"
        raise FileNotFoundError(msg)

    with open(base_path, encoding="utf-8") as f:  # noqa: PTH123
        text = f.read()

    # Limpeza simples de Markdown
    text = re.sub(r"#.*\n", "", text)  # remove títulos
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)  # remove negrito
    text = re.sub(r"\*(.*?)\*", r"\1", text)  # remove itálico
    text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)  # remove links
    return text.replace("\n", " ")  # opcional: transforma quebras de linha em espaço


# ------------------------------
# TOKENIZAÇÃO E INDICE BM25
# ------------------------------
def create_bm25_index(text):  # noqa: ANN001, ANN201
    """Cria índice BM25 a partir do texto"""
    # Tokeniza por sentenças ou por parágrafos
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    tokenized = [s.split() for s in sentences]
    bm25 = BM25Okapi(tokenized)
    return bm25, sentences


# ------------------------------
# FUNÇÃO DE CONSULTA BM25
# ------------------------------
def query_bm25(bm25, sentences, query, top_k=5):  # noqa: ANN001, ANN201
    """Retorna os top_k trechos mais relevantes para a query"""
    query_tokens = query.split()
    scores = bm25.get_scores(query_tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
        :top_k
    ]
    return [sentences[i] for i in top_indices]


# ------------------------------
# TESTE LOCAL
# ------------------------------
if __name__ == "__main__":
    print("📄 Carregando Constituição...")
    constitution_text = load_constitution()
    bm25, sentences = create_bm25_index(constitution_text)

    pergunta = "O que a Constituição diz sobre direitos fundamentais?"
    top_matches = query_bm25(bm25, sentences, pergunta, top_k=5)

    print("\n🧾 Pergunta:", pergunta)
    print("\n💬 Resposta BM25 (trechos mais relevantes):")
    for i, match in enumerate(top_matches, 1):
        print(f"{i}. {match}")
