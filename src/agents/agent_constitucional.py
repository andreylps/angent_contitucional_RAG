# src/agents/agent_constitucional.py

import json
import os

from dotenv import load_dotenv
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.prompts import PromptTemplate
from langchain_classic.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings

from parsers.parse_constitution import OUTPUT_FILE  # jsonl processado
from pipelines.weighted_hybrid import WeightedHybridRetriever
from utils.llm_utils import load_llm

# ---------------------------------------------------------------------
# 🎯 1. Carregar variáveis de ambiente
# ---------------------------------------------------------------------
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    msg = "⚠️ OPENAI_API_KEY não definido no .env"
    raise ValueError(msg)

# 🔑 Inicializa embeddings (usando OpenAI)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# ---------------------------------------------------------------------
# 🎯 2. Inicializar LLM e memória
# ---------------------------------------------------------------------
llm = load_llm()
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# ---------------------------------------------------------------------
# 🎯 3. Template de prompt jurídico
# ---------------------------------------------------------------------
prompt_template = """
Você é um assistente jurídico constitucionalista, especialista na Constituição Federal do Brasil.
Responda de forma técnica, mas compreensível, sempre citando os artigos e incisos relevantes.

Histórico da conversa:
{chat_history}

Contexto dos documentos:
{context}

Pergunta atual:
{question}

Se a resposta não estiver diretamente na Constituição, informe isso ao usuário e explique brevemente o motivo.
Inclua referências legais conforme disponíveis.
"""  # noqa: E501

prompt = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template=prompt_template,
)


# ---------------------------------------------------------------------
# 🎯 4. Carregar parágrafos do JSONL
# ---------------------------------------------------------------------
def carregar_paragrafos() -> list[str]:
    paragrafos = []
    if not os.path.exists(OUTPUT_FILE):  # noqa: PTH110
        msg = f"Arquivo {OUTPUT_FILE} não encontrado"
        raise FileNotFoundError(msg)

    with open(OUTPUT_FILE, encoding="utf-8") as f:  # noqa: PTH123
        for line in f:
            item = json.loads(line)
            paragrafos.append(item["content"])
    return paragrafos


# ---------------------------------------------------------------------
# 🎯 5. Inicializar Chroma VectorStore
# ---------------------------------------------------------------------

# Carrega base vetorial
try:
    chroma = Chroma(
        persist_directory="./chroma_db_constitucional", embedding_function=embeddings
    )
    vector_retriever = chroma.as_retriever(search_kwargs={"k": 10})
    print("✅ Chroma carregado do diretório persistente")
except Exception as e:  # noqa: BLE001
    print(f"⚠️ Criando novo Chroma: {e}")
    # Se não existir, cria novo
    paragrafos = carregar_paragrafos()
    vector_store = Chroma.from_texts(
        texts=paragrafos,
        embedding=embeddings,
        collection_name="constitution",
        metadatas=[{"source": "constituição"} for _ in paragrafos],
        persist_directory="./chroma_db_constitucional",
    )
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": 10})
    print("✅ Novo Chroma criado e persistido")

# ---------------------------------------------------------------------
# 🎯 6. Inicializar BM25 Retriever
# ---------------------------------------------------------------------
from utils.carrega_constituicao import carregar_constituicao  # noqa: E402

try:
    paragrafos = carregar_paragrafos()
except FileNotFoundError:
    print("⚠️ Usando carregar_constituicao() como fallback")
    paragrafos = carregar_constituicao()

# 📚 BM25 retriever (baseado no texto dos parágrafos da Constituição)
bm25_retriever = BM25Retriever.from_texts(paragrafos)

# ---------------------------------------------------------------------
# 🎯 7. Configurar Retriever Híbrido Ponderado
# ---------------------------------------------------------------------
# ⚖️ Híbrido ponderado
weighted_retriever = WeightedHybridRetriever(
    bm25_retriever=bm25_retriever,
    vector_retriever=vector_retriever,
    weight_bm25=0.4,
    weight_vector=0.6,
    top_k=10,
)

# 🤖 Modelo de linguagem
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ---------------------------------------------------------------------
# 🎯 8. Conversational Retrieval Chain
# ---------------------------------------------------------------------
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=weighted_retriever,
    memory=memory,
    verbose=False,  # ativa e habilita o debug
    combine_docs_chain_kwargs={"prompt": prompt},
)

print("🤖 Agente Constitucional carregado com sucesso!")


# ---------------------------------------------------------------------
# 🎯 9. Função principal para consulta
# ---------------------------------------------------------------------
def consultar_constituicao(pergunta: str) -> str:
    """
    Consulta a Constituição Federal usando o agente jurídico.

    Args:
        pergunta: Pergunta sobre a Constituição

    Returns:
        Resposta jurídica baseada na Constituição
    """
    try:
        resposta = qa_chain.invoke({"question": pergunta})
        return resposta["answer"]
    except Exception as e:  # noqa: BLE001
        return f"❌ Erro ao consultar a Constituição: {e!s}"


# ---------------------------------------------------------------------
# 🚀 10. Execução direta (modo teste)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("🧠 Agente Constitucional iniciado. Digite 'sair' para encerrar.\n")

    while True:
        try:
            pergunta = input("❓ Pergunta: ").strip()

            if pergunta.lower() in ["sair", "exit", "quit"]:
                print("👋 Encerrando agente constitucional...")
                break

            if not pergunta:
                print("⚠️ Por favor, digite uma pergunta válida.\n")
                continue

            resposta = consultar_constituicao(pergunta)
            print(f"\n⚖️ Resposta:\n{resposta}\n")
            print("-" * 80 + "\n")

        except KeyboardInterrupt:
            print("\n👋 Encerrando agente constitucional...")
            break
        except Exception as e:  # noqa: BLE001
            print(f"❌ Erro inesperado: {e!s}\n")
