# src/pipelines/rag_retriever.py

import os

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from openai import OpenAI
from pinecone import Pinecone

# ------------------------------
# CONFIGURAÇÃO DE AMBIENTE
# ------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "constitution-index")

# ------------------------------
# INICIALIZAÇÃO DO PINECONE
# ------------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)


# ------------------------------
# DETECÇÃO AUTOMÁTICA DE DIMENSÃO
# ------------------------------
def get_index_dimension():  # noqa: ANN201
    try:
        info = pc.describe_index(PINECONE_INDEX)
        return info.dimension  # noqa: TRY300
    except Exception as e:  # noqa: BLE001
        print("⚠️ Erro ao obter dimensão do índice:", e)
        return 1536  # fallback padrão


index_dim = get_index_dimension()
print(f"📏 Dimensão do índice Pinecone: {index_dim}")

# Define automaticamente o modelo de embedding
embedding_model = (
    "text-embedding-3-large" if index_dim == 3072 else "text-embedding-3-small"  # noqa: PLR2004
)
print(f"🧠 Modelo de embedding selecionado: {embedding_model}")

# ------------------------------
# EMBEDDING FUNCTION
# ------------------------------
openai_client = OpenAI(api_key=OPENAI_API_KEY)


def embed_text(text: str):  # noqa: ANN201
    response = openai_client.embeddings.create(model=embedding_model, input=text)
    return response.data[0].embedding


# ------------------------------
# BUSCA NO VETORSTORE
# ------------------------------
def vectorstore_search(query, top_k=5):  # noqa: ANN001, ANN201
    query_vector = embed_text(query)
    result = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
    return result.matches  # type: ignore  # noqa: PGH003


# ------------------------------
# AGENTE RAG (LLM)
# ------------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)  # type: ignore  # noqa: PGH003

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


# ------------------------------
# FUNÇÃO DE CONSULTA RAG
# ------------------------------
def query_rag(user_query, k=5):  # noqa: ANN001, ANN201
    docs = vectorstore_search(user_query, top_k=k)
    context_text = "\n\n".join(
        [
            match.metadata.get("content", "")
            for match in docs
            if "content" in match.metadata
        ]
    )

    prompt = prompt_template.format(context=context_text, question=user_query)
    response = llm.invoke(prompt)
    return response.content


# ------------------------------
# TESTE LOCAL
# ------------------------------
if __name__ == "__main__":
    pergunta = "O que a Constituição diz sobre direitos fundamentais?"
    resposta = query_rag(pergunta)
    print("\n🧾 Pergunta:", pergunta)
    print("\n💬 Resposta:", resposta)
