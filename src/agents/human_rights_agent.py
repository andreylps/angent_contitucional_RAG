import asyncio
from typing import Any

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable

from ..utils.cross_encoder_reranker import CrossEncoderReRanker

# Template de prompt para gerar variações da pergunta original (reutilizado)
MULTI_QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""Você é um assistente de IA especialista em direito. Sua tarefa é gerar 3 versões diferentes da pergunta do usuário para recuperar documentos relevantes de um banco de dados vetorial.

Ao gerar múltiplas perspectivas sobre a pergunta do usuário, seu objetivo é ajudar o usuário a superar algumas das limitações da busca por similaridade baseada em distância.

Forneça as perguntas alternativas separadas por quebras de linha. Não numere as perguntas.

Pergunta Original: {question}""",
)


class HumanRightsAgent:
    """Agente especializado em Direitos Humanos com busca Multi-Query."""

    def __init__(self, domain: str, retriever: Any, llm: Runnable):
        self.name = "human_rights_agent"
        self.domain = domain
        self.retriever = retriever
        self.llm = llm
        self.final_answer_prompt = self._create_final_answer_prompt()
        self.reranker = CrossEncoderReRanker()

    def _create_final_answer_prompt(self) -> PromptTemplate:
        """Cria o prompt final para gerar a resposta com base no contexto."""
        return PromptTemplate(
            input_variables=["context", "question"],
            template="""Você é um assistente especialista em Direitos Humanos, com foco na Convenção Americana de Direitos Humanos (CADH). Sua tarefa é responder à pergunta do usuário de forma clara, concisa e bem estruturada, baseando-se exclusivamente nos trechos de documentos fornecidos no contexto.

**Instruções Importantes:**
1.  **Sintetize a Informação:** Os documentos no contexto são fragmentos. Sua principal tarefa é conectar as informações de múltiplos fragmentos para construir uma resposta completa.
2.  **Seja Exclusivo:** Responda APENAS com base no contexto. Não utilize nenhum conhecimento prévio.
3.  **Seja Honesto:** Se, após analisar todos os fragmentos, a informação para responder à pergunta não estiver presente, informe que não foi possível encontrar uma resposta conclusiva nos documentos consultados.

Contexto:
{context}

Pergunta: {question}

Resposta:""",
        )

    async def _generate_queries(self, question: str) -> list[str]:
        """Gera perguntas alternativas usando o LLM."""
        # Reutiliza a mesma lógica dos outros agentes
        try:
            generate_queries_chain = MULTI_QUERY_PROMPT | self.llm | StrOutputParser()
            response = await generate_queries_chain.ainvoke({"question": question})
            all_queries = [question] + [
                q.strip() for q in response.split("\n") if q.strip()
            ]
            return all_queries
        except Exception:
            return [question]

    async def _get_unique_documents(self, queries: list[str]) -> list[Document]:
        """Busca documentos para múltiplas perguntas e remove duplicatas."""
        # Reutiliza a mesma lógica dos outros agentes
        tasks = [self.retriever.ainvoke(query) for query in queries]
        results = await asyncio.gather(*tasks)
        all_docs = [doc for sublist in results for doc in sublist]
        unique_docs_map = {
            (doc.page_content, doc.metadata.get("source", "")): doc for doc in all_docs
        }
        return list(unique_docs_map.values())

    async def _rerank_documents(
        self, query: str, documents: list[Document]
    ) -> list[Document]:
        """Usa o LLM para reordenar e selecionar os documentos mais relevantes."""
        # Reutiliza a mesma lógica dos outros agentes
        if not documents:
            return []
        doc_texts = [
            f"ID do Documento: [{i}]\nConteúdo: {doc.page_content}"
            for i, doc in enumerate(documents)
        ]
        formatted_docs = "\n\n---\n\n".join(doc_texts)
        rerank_prompt = PromptTemplate.from_template(
            """Você é um assistente de IA especialista em análise de relevância. Sua tarefa é analisar uma lista de documentos e uma pergunta, e retornar os IDs dos 4 documentos mais relevantes para responder à pergunta.

Documentos:
{documents}

Pergunta: {question}

Responda APENAS com uma lista de IDs dos 4 documentos mais relevantes, separados por vírgula. Exemplo: [0], [3], [1], [8]"""
        )
        rerank_chain = rerank_prompt | self.llm | StrOutputParser()
        response = await rerank_chain.ainvoke(
            {"documents": formatted_docs, "question": query}
        )
        try:
            relevant_ids = [int(id_str.strip("[] ")) for id_str in response.split(",")]
            return [documents[i] for i in relevant_ids if i < len(documents)]
        except (ValueError, IndexError):
            return documents[:4]

    async def invoke(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Invoca o agente com a lógica Multi-Query e Re-ranking."""
        query = inputs["query"]

        queries = await self._generate_queries(query)
        documents = await self._get_unique_documents(queries)
        final_documents = await self._rerank_documents(query, documents)

        if not final_documents:
            return {
                "agent": self.name,
                "agent_domain": self.domain,
                "answer": "Não foram encontrados documentos relevantes para responder a esta pergunta.",
                "sources": [],
                "confidence": 0.1,
                "status": "no_documents",
            }

        context = "\n\n---\n\n".join([doc.page_content for doc in final_documents])
        final_chain = self.final_answer_prompt | self.llm | StrOutputParser()
        answer = await final_chain.ainvoke({"context": context, "question": query})
        sources = list(
            set(doc.metadata.get("file_name", "N/A") for doc in final_documents)
        )

        return {
            "agent": self.name,
            "agent_domain": self.domain,
            "answer": answer,
            "sources": sources,
            "confidence": 0.75,
            "status": "success",
        }
