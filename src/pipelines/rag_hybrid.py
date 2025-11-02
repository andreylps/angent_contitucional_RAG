# src/pipelines/rag_hybrid.py

import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from openai import OpenAI
from pinecone import Pinecone
from rank_bm25 import BM25Okapi

load_dotenv()
# ------------------------------
# CONFIGURAÇÃO DE AMBIENTE
# ------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "constitution-index")

# ------------------------------
# INICIALIZAÇÃO PINECONE
# ------------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

# ------------------------------
# INICIALIZAÇÃO OPENAI
# ------------------------------
openai_client = OpenAI(api_key=OPENAI_API_KEY)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)  # type: ignore  # noqa: PGH003


# ------------------------------
# CARREGANDO DOCUMENTOS PARA BM25
# ------------------------------
def load_constitution_texts(base_path: str) -> list[str]:
    """Carrega a Constituição e retorna lista de parágrafos."""
    if not os.path.exists(base_path):  # noqa: PTH110
        msg = f"Arquivo não encontrado: {base_path}"
        raise FileNotFoundError(msg)

    with open(base_path, encoding="utf-8") as f:  # noqa: PTH123
        content = f.read()

    # Separar em parágrafos ou blocos
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    print(f"📄 {len(paragraphs)} parágrafos carregados para BM25")
    return paragraphs


BASE_PATH = os.path.abspath(  # noqa: PTH100
    os.path.join(  # noqa: PTH118
        os.path.dirname(__file__),  # noqa: PTH120
        "..",
        "..",
        "data",
        "constitution",
        "constituicao.md",
    )
)

print(f"📄 Caminho da Constituição: {BASE_PATH}")

documents = load_constitution_texts(BASE_PATH)

# Tokenizar para BM25
tokenized_corpus = [doc.split(" ") for doc in documents]
bm25 = BM25Okapi(tokenized_corpus)


# ------------------------------
# FUNÇÃO DE EMBEDDING
# ------------------------------
def embed_text(text: str, model="text-embedding-3-small"):  # noqa: ANN001, ANN201
    response = openai_client.embeddings.create(model=model, input=text)
    return response.data[0].embedding


# ------------------------------
# BUSCA HÍBRIDA
# ------------------------------
def hybrid_search(query: str, top_k=5):  # noqa: ANN001, ANN201
    """
    Realiza busca híbrida: BM25 + Pinecone.
    Retorna lista de textos relevantes.
    """
    # --- BM25 ---
    query_tokens = query.split(" ")
    bm25_scores = bm25.get_scores(query_tokens)
    top_indices_bm25 = bm25_scores.argsort()[::-1][:top_k]
    bm25_docs = [documents[i] for i in top_indices_bm25]

    # --- Pinecone ---
    query_vector = embed_text(query)
    pinecone_results = index.query(
        vector=query_vector, top_k=top_k, include_metadata=True
    )
    pinecone_docs = [
        match.metadata.get("content", "")
        for match in pinecone_results.matches  # type: ignore  # noqa: PGH003
        if "content" in match.metadata
    ]

    # --- Combinar resultados ---
    return bm25_docs + pinecone_docs


# ------------------------------
# AGENTE HÍBRIDO (RAG)
# ------------------------------
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Você é um assistente jurídico especializado na Constituição Brasileira.
Use o contexto abaixo para responder à pergunta do usuário de forma clara e precisa.

Contexto:
{context}

Pergunta:
{question}

Resposta:
""",
)


def query_rag_hybrid(user_query: str, top_k=5):  # noqa: ANN001, ANN201
    docs = hybrid_search(user_query, top_k=top_k)
    context_text = "\n\n".join(docs)
    prompt = prompt_template.format(context=context_text, question=user_query)
    response = llm.invoke(prompt)
    return response.content


# ------------------------------
# TESTE LOCAL
# ------------------------------
if __name__ == "__main__":
    pergunta = "O que a Constituição diz sobre direitos fundamentais?"
    resposta = query_rag_hybrid(pergunta)
    print("\n🧾 Pergunta:", pergunta)
    print("\n💬 Resposta:", resposta)
