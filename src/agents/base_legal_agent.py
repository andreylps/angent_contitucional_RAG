#!/usr/bin/env python3
"""
Classe base para todos os agentes jurídicos especializados
"""

from abc import ABC, abstractmethod
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.retrievers import BaseRetriever
from langchain_openai import ChatOpenAI


class BaseLegalAgent(ABC):
    """Classe abstrata base para agentes jurídicos especializados"""

    def __init__(
        self, name: str, retriever: BaseRetriever, llm: ChatOpenAI, system_prompt: str
    ) -> None:
        self.name = name
        self.retriever = retriever
        self.llm = llm
        self.system_prompt = system_prompt

        # Cria o prompt template
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        )

    @abstractmethod
    def get_domain(self) -> str:
        """Retorna o domínio jurídico do agente"""

    def invoke(
        self, query: str, conversation_history: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        """
        Executa o agente com uma consulta e retorna a resposta

        Args:
            query: A pergunta do usuário
            conversation_history: Histórico da conversa (opcional)

        Returns:
            Dict com resposta e metadados
        """
        try:
            # Recupera documentos relevantes - MÉTODO CORRIGIDO
            docs = self.retriever.invoke(query)

            # Calcula confiança baseada nos documentos
            confidence = self._calculate_confidence(query, docs)

            if not docs or confidence < 0.1:  # noqa: PLR2004
                return {
                    "agent": self.name,
                    "domain": self.get_domain(),
                    "answer": "Não encontrei informações suficientes para responder esta pergunta na minha base de conhecimento especializada.",  # noqa: E501
                    "sources": [],
                    "confidence": confidence,
                    "status": "no_relevant_docs",
                }

            # Prepara o contexto dos documentos
            context = self._format_documents_context(docs)

            # Prepara mensagens para o LLM
            messages = self._prepare_messages(query, context, conversation_history)

            # Gera resposta
            response = self.llm.invoke(messages)
            answer = response.content

            return {
                "agent": self.name,
                "domain": self.get_domain(),
                "answer": answer,
                "sources": [doc.metadata for doc in docs],
                "confidence": confidence,
                "status": "success",
            }

        except Exception as e:  # noqa: BLE001
            return {
                "agent": self.name,
                "domain": self.get_domain(),
                "answer": f"Erro ao processar a consulta: {e!s}",
                "sources": [],
                "confidence": 0.0,
                "status": "error",
            }

    def _prepare_messages(
        self,
        query: str,
        context: str,
        conversation_history: list[dict[str, Any]] | None = None,
    ) -> list[BaseMessage]:
        """Prepara as mensagens para o LLM"""
        messages = []

        # Adiciona histórico da conversa se disponível
        if conversation_history:
            for msg in conversation_history[-6:]:  # Últimas 6 mensagens
                if msg.get("role") == "user":
                    messages.append(HumanMessage(content=msg.get("content", "")))
                elif msg.get("role") == "assistant":
                    messages.append(SystemMessage(content=msg.get("content", "")))

        # Adiciona contexto e pergunta atual
        full_query = f"Contexto:\n{context}\n\nPergunta: {query}"
        messages.append(HumanMessage(content=full_query))

        return messages

    def _format_documents_context(self, docs: list[Any]) -> str:
        """Formata os documentos como contexto para o LLM"""
        context_parts = []

        for i, doc in enumerate(docs, 1):
            source_info = f"Fonte: {doc.metadata.get('source', 'Desconhecida')}"
            if doc.metadata.get("page"):
                source_info += f" - Página {doc.metadata['page']}"

            context_parts.append(f"--- Documento {i} ---")
            context_parts.append(source_info)
            context_parts.append(f"Conteúdo: {doc.page_content}")
            context_parts.append("")

        return "\n".join(context_parts)

    def _calculate_confidence(self, query: str, docs: list[Any]) -> float:
        """
        Calcula a confiança da resposta baseada na query e documentos

        Args:
            query: A pergunta do usuário
            docs: Documentos recuperados

        Returns:
            Valor de confiança entre 0.0 e 1.0
        """
        if not docs:
            return 0.0

        # Confiança base baseada no número de documentos
        base_confidence = min(len(docs) / 5.0, 1.0)

        # Boost baseado na relevância (simplificado)
        query_terms = query.lower().split()
        relevance_boost = 0.0

        for doc in docs[:3]:  # Considera apenas os 3 primeiros docs
            content_lower = doc.page_content.lower()
            matches = sum(1 for term in query_terms if term in content_lower)
            relevance_boost += matches * 0.05

        return min(base_confidence + relevance_boost, 1.0)
