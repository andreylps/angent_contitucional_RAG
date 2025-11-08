#!/usr/bin/env python3
"""
Classe base para todos os agentes jurídicos especializados
"""

from abc import ABC
from typing import Any

import numpy as np
from langchain_core.messages import (  # type: ignore  # noqa: PGH003
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.prompts import (  # type: ignore  # noqa: PGH003
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.retrievers import BaseRetriever  # type: ignore  # noqa: PGH003
from langchain_openai import ChatOpenAI  # type: ignore  # noqa: PGH003
from sklearn.metrics.pairwise import cosine_similarity

from ..utils.logger import logger  # Importa o logger  # noqa: TID252


class BaseLegalAgent(ABC):  # noqa: B024
    """Classe abstrata base para agentes jurídicos especializados"""

    def __init__(
        self,
        name: str,
        domain: str,  # ✅ 1. Adiciona o parâmetro 'domain'
        retriever: BaseRetriever,
        llm: ChatOpenAI,
        system_prompt: str,
    ) -> None:
        self.name = name
        self.domain = domain  # ✅ 2. Armazena o domínio
        self.retriever = retriever
        self.llm = llm
        self.system_prompt = system_prompt

        # ✅ ETAPA 1.2: Inicializa o modelo de embeddings para cálculo de confiança
        # Usamos um modelo separado para não interferir com o retriever, se necessário
        # ✅ OTIMIZAÇÃO E ATUALIZAÇÃO: Usa o modelo local do novo pacote langchain-huggingface.  # noqa: E501
        from langchain_community.embeddings import (
            HuggingFaceEmbeddings,  # noqa: PLC0415
        )

        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )

        # Cria o prompt template
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        )

    def get_domain(self) -> str:
        return self.domain

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
            confidence = self._calculate_confidence_with_similarity(query, docs)

            if not docs or confidence < 0.1:  # noqa: PLR2004
                return {
                    "agent": self.name,
                    "agent_domain": self.get_domain(),
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
                "agent_domain": self.get_domain(),
                "answer": answer,
                "sources": [doc.metadata for doc in docs],
                "confidence": confidence,
                "status": "success",
            }

        except Exception as e:  # noqa: BLE001
            return {
                "agent": self.name,
                "agent_domain": self.get_domain(),
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
            # ✅ CORREÇÃO FINAL: Garante que doc.metadata é um dicionário antes de acessar.  # noqa: E501
            # Se não for, usa um dicionário vazio para evitar AttributeErrors ou TypeErrors.  # noqa: E501
            metadata = doc.metadata
            if not isinstance(metadata, dict):
                logger.warning(
                    f"Metadados do documento não são um dicionário: {type(metadata)}. Usando metadados vazios."  # noqa: E501
                )
                metadata = {}

            source_info = f"Fonte: {metadata.get('source', 'Desconhecida')}"
            if metadata.get("page"):
                source_info += f" - Página {metadata.get('page')}"
            if metadata.get("domain"):
                source_info += f" - Domínio: {metadata.get('domain')}"

            context_parts.append(f"--- Documento {i} ---")
            context_parts.append(source_info)
            context_parts.append(f"Conteúdo: {doc.page_content}")
            context_parts.append("")

        return "\n".join(context_parts)

    def _calculate_confidence_with_similarity(
        self, query: str, docs: list[Any]
    ) -> float:
        """
        Calcula a confiança da resposta usando similaridade de cosseno entre
        a query e os documentos recuperados.

        Args:
            query: A pergunta do usuário
            docs: Documentos recuperados

        Returns:
            A média da similaridade de cosseno dos documentos, resultando
            em um score de confiança entre 0.0 e 1.0.
        """
        if not docs:
            return 0.0

        try:
            # 1. Cria o embedding da query do usuário
            # ✅ CORREÇÃO: HuggingFaceEmbeddings usa embed_documents.
            # Passamos a query como uma lista de um item e pegamos o primeiro resultado.
            query_embedding = self.embedding_model.embed_documents([query])[0]

            # 2. Cria os embeddings dos conteúdos dos documentos recuperados
            doc_contents = [doc.page_content for doc in docs]
            doc_embeddings = self.embedding_model.embed_documents(doc_contents)

            # 3. Calcula a similaridade de cosseno - ✅ CORREÇÃO: Converte para arrays NumPy  # noqa: E501
            query_embedding_np = np.array(query_embedding).reshape(1, -1)
            doc_embeddings_np = np.array(doc_embeddings)

            similarities = cosine_similarity(query_embedding_np, doc_embeddings_np)[0]

            # 4. A confiança é a média das similaridades
            average_similarity = np.mean(similarities)

            return float(average_similarity)
        except Exception:  # noqa: BLE001
            # Em caso de erro no cálculo (ex: API), retorna uma confiança baixa
            return 0.1
