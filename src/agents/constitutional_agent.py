import asyncio
from typing import Any

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable

from ..utils.cross_encoder_reranker import CrossEncoderReRanker

# Template de prompt para gerar variações da pergunta original
MULTI_QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""Você é um assistente de IA especialista em direito. Sua tarefa é gerar 3 versões diferentes da pergunta do usuário para recuperar documentos relevantes de um banco de dados vetorial.

Ao gerar múltiplas perspectivas sobre a pergunta do usuário, seu objetivo é ajudar o usuário a superar algumas das limitações da busca por similaridade baseada em distância.

Forneça as perguntas alternativas separadas por quebras de linha. Não numere as perguntas.

Pergunta Original: {question}""",
)


class ConstitutionalAgent:
    """Agente especializado em Direito Constitucional com busca Multi-Query."""

    def __init__(self, domain: str, retriever: Any, llm: Runnable):
        self.name = "constitutional_agent"
        self.domain = domain
        self.retriever = retriever
        self.llm = llm
        self.final_answer_prompt = self._create_final_answer_prompt()
        self.reranker = CrossEncoderReRanker()

    def _create_final_answer_prompt(self) -> PromptTemplate:
        """Cria o prompt final para gerar a resposta com base no contexto."""
        return PromptTemplate(
            input_variables=["context", "question", "chat_history"],
            template="""Você é um assistente especialista em Direito Constitucional brasileiro. Sua tarefa é responder à pergunta do usuário de forma clara e concisa, baseando-se exclusivamente nos trechos de documentos fornecidos no contexto e considerando o histórico da conversa.

**Instruções Importantes:**
1.  **Foque na Pergunta Atual:** Use o histórico da conversa para entender o contexto, mas sua resposta deve focar em responder diretamente à última pergunta do usuário.
1.  **Sintetize a Informação:** Os documentos no contexto são fragmentos. Sua principal tarefa é conectar as informações de múltiplos fragmentos para construir uma resposta completa.
2.  **Seja Exclusivo:** Responda APENAS com base no contexto. Não utilize nenhum conhecimento prévio.
3.  **Seja Honesto:** Se, após analisar todos os fragmentos, a informação para responder à pergunta não estiver presente, informe que não foi possível encontrar uma resposta conclusiva nos documentos consultados.

Contexto:
{context}

Histórico da Conversa:
{chat_history}

Pergunta: {question}

Resposta:""",
        )

    async def _generate_queries(self, question: str) -> list[str]:
        """Gera perguntas alternativas usando o LLM."""
        try:
            print(f"   [{self.name}] Gerando variações da pergunta...")
            generate_queries_chain = MULTI_QUERY_PROMPT | self.llm | StrOutputParser()
            response = await generate_queries_chain.ainvoke({"question": question})

            # Adiciona a pergunta original e as geradas, filtrando linhas vazias
            all_queries = [question] + [
                q.strip() for q in response.split("\n") if q.strip()
            ]
            print(f"   [{self.name}] Total de perguntas para busca: {len(all_queries)}")
            return all_queries
        except Exception as e:
            print(
                f"   ⚠️ [{self.name}] Erro ao gerar perguntas: {e}. Usando apenas a pergunta original."
            )
            return [question]

    async def _get_unique_documents(self, queries: list[str]) -> list[Document]:
        """Busca documentos para múltiplas perguntas e remove duplicatas."""
        print(f"   [{self.name}] Executando busca com {len(queries)} perguntas...")
        # Cria tarefas para buscar documentos para cada pergunta em paralelo
        tasks = [self.retriever.ainvoke(query) for query in queries]
        # Aguarda todas as buscas
        results = await asyncio.gather(*tasks)

        # Junta todos os documentos de todas as buscas
        all_docs = [doc for sublist in results for doc in sublist]

        # Remove documentos duplicados usando o conteúdo e a fonte como chave
        unique_docs_map = {}
        for doc in all_docs:
            key = (doc.page_content, doc.metadata.get("source", ""))
            if key not in unique_docs_map:
                unique_docs_map[key] = doc

        unique_docs = list(unique_docs_map.values())
        print(f"   [{self.name}] Documentos únicos encontrados: {len(unique_docs)}")
        return unique_docs

    async def _rerank_documents(
        self, query: str, documents: list[Document]
    ) -> list[Document]:
        """
        ✅ v0.2: Usa o CrossEncoder para reordenar os documentos.
        Esta função é executada em um thread separado para não bloquear o loop de eventos.
        """
        if not documents:
            return []

        print(
            f"   [{self.name}] Reordenando {len(documents)} documentos com CrossEncoder..."
        )

        loop = asyncio.get_event_loop()
        # A lógica de re-ranking agora é chamada diretamente no executor.
        reranked_docs: list[Document] = await loop.run_in_executor(  # type: ignore
            None, self.reranker.rerank, query, documents
        )
        print(
            f"   [{self.name}] Documentos reordenados e selecionados: {len(reranked_docs)}"
        )
        return reranked_docs

    async def invoke(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """
        Invoca o agente com a lógica Multi-Query.
        """
        # ✅ v0.3.1: O agente agora recebe uma pergunta já reescrita e autônoma.
        # A pergunta original é mantida para a resposta final.
        standalone_query = inputs["query"]
        original_query = inputs.get("original_query", standalone_query)
        conversation_history = inputs.get("conversation_history", [])

        print(f"   🚀 Agente '{self.name}' invocado para o domínio '{self.domain}'")

        # 1. Gerar múltiplas perguntas (usando a pergunta autônoma)
        queries = await self._generate_queries(standalone_query)

        # 2. Recuperar documentos únicos usando todas as perguntas
        documents = await self._get_unique_documents(queries)

        # 3. ✅ v0.2: Reordena e seleciona os melhores documentos com CrossEncoder
        final_documents = await self._rerank_documents(standalone_query, documents)

        if not final_documents:
            return {
                "agent": self.name,
                "agent_domain": self.domain,
                "answer": "Não foram encontrados documentos relevantes para responder a esta pergunta.",
                "sources": [],
                "confidence": 0.1,
                "status": "no_documents",
            }

        # 4. Formatar o contexto e gerar a resposta final
        context = "\n\n---\n\n".join([doc.page_content for doc in final_documents])

        final_chain = self.final_answer_prompt | self.llm | StrOutputParser()

        # Formata o histórico para o prompt final
        formatted_history = "\n".join(
            [f"{msg['role']}: {msg['content']}" for msg in conversation_history]
        )

        answer = await final_chain.ainvoke(
            {
                "context": context,
                "question": original_query,
                "chat_history": formatted_history,
            }
        )

        # Extrai as fontes dos documentos
        sources = list(
            set(doc.metadata.get("file_name", "N/A") for doc in final_documents)
        )

        return {
            "agent": self.name,
            "agent_domain": self.domain,
            "answer": answer,
            "sources": sources,
            "confidence": 0.75,  # Valor placeholder, podemos refinar depois
            "status": "success",
        }
