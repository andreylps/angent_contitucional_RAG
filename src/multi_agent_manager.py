#!/usr/bin/env python3
"""
Gerenciador principal do sistema multiagente RAG jurídico
"""

import asyncio
import os
import time
import uuid
from datetime import UTC, datetime
from typing import Any

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# Carrega variáveis do .env
# Carrega variáveis do .env
load_dotenv()

# ✅ CORREÇÃO: Imports relativos sem "src."
from .agents.constitutional_agent import ConstitutionalAgent  # noqa: E402
from .agents.consumer_agent import ConsumerAgent  # noqa: E402
from .agents.human_rights_agent import HumanRightsAgent  # noqa: E402
from .agents.router_agent import LegalRouterAgent  # noqa: E402
from .pipelines.specialized_retrievers import (  # noqa: E402
    create_specialized_retriever,
)
from .utils.interaction_logger import log_interaction  # ✅ v0.4: Importa o logger
from .utils.web_search_tool import (
    WebSearchTool,  # ✅ v0.5: Importa a ferramenta de busca
)

# ✅ v0.3.1: Template para re-escrever a pergunta com base no histórico (movido para o Manager)
REWRITE_QUERY_PROMPT = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="""Dada a conversa a seguir e uma pergunta de acompanhamento, reformule a pergunta de acompanhamento para ser uma pergunta independente que possa ser entendida sem o histórico.

Histórico da Conversa:
{chat_history}

Pergunta de Acompanhamento: {question}

Pergunta Independente:""",
)

# ✅ v0.5: Template para responder com base em busca na web (movido para o Manager)
WEB_ANSWER_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""Você é um assistente de IA. Sua tarefa é responder à pergunta do usuário com base nos trechos de páginas da web fornecidos no contexto.

**Instruções OBRIGATÓRIAS:**
1.  **Baseie-se nos Fatos:** Responda APENAS com base no contexto fornecido (trechos da web). Não use conhecimento prévio.
2.  **Cite a Fonte:** Ao final de cada informação relevante, cite a URL da fonte usando o formato `(Fonte: [URL])`.
3.  **Adicione um Aviso:** Ao final da sua resposta, inclua o seguinte aviso, exatamente como está escrito:
    "---
    **Aviso:** Esta resposta foi gerada com base em informações de fontes externas da web e não da base de conhecimento jurídica interna. Recomenda-se a validação das informações na fonte original."

Contexto da Web:
{context}

Pergunta: {question}

Resposta:""",
)


class MultiAgentManager:
    """Gerenciador principal que coordena todos os agentes especializados"""

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        # ✅ Verifica se API key existe no .env
        openai_api_key = os.getenv("OPENAI_API_KEY")

        if not openai_api_key:
            msg = "OPENAI_API_KEY não encontrada no arquivo .env"
            raise ValueError(msg)

        # ✅ CORREÇÃO: ChatOpenAI pega API_KEY automaticamente do ambiente
        self.llm = ChatOpenAI(model=model, temperature=0.1)

        # Inicializa o router
        self.router = LegalRouterAgent(self.llm)

        # Inicializa os agentes especializados
        self.agents: dict[str, Any] = self._initialize_agents()

        # ✅ v0.3: Adiciona um estado para armazenar o histórico da conversa
        self.conversation_history: list[dict[str, Any]] = []

        # ✅ v0.5: Inicializa a ferramenta de busca na web e o prompt de resposta
        self.web_search_tool = WebSearchTool()
        self.web_answer_prompt = WEB_ANSWER_PROMPT

        print("✅ MultiAgentManager inicializado com sucesso!")
        print("   - Router Agent: Pronto")
        print(f"   - Agentes especializados: {len(self.agents)} carregados")
        print(f"   - LLM: {model}")
        print(
            f"   - API Key: {'✅ Carregada' if openai_api_key else '❌ Não encontrada'}"
        )

    def _initialize_agents(self) -> dict[str, Any]:
        """Inicializa todos os agentes especializados com seus retrievers"""
        agents: dict[str, Any] = {}

        try:
            # Agente Constitucional
            constitutional_retriever = create_specialized_retriever(
                "constitutional_law"
            )
            agents["constitutional_law"] = ConstitutionalAgent(
                domain="constitutional_law",
                retriever=constitutional_retriever,
                llm=self.llm,
            )
            print("   ✅ ConstitutionalAgent carregado")

            # Agente Consumer
            consumer_retriever = create_specialized_retriever("consumer_law")
            agents["consumer_law"] = ConsumerAgent(
                domain="consumer_law", retriever=consumer_retriever, llm=self.llm
            )
            print("   ✅ ConsumerAgent carregado")

            # Agente Human Rights
            human_rights_retriever = create_specialized_retriever("human_rights_law")
            agents["human_rights_law"] = HumanRightsAgent(
                domain="human_rights_law",
                retriever=human_rights_retriever,
                llm=self.llm,
            )
            print("   ✅ HumanRightsAgent carregado")

        except Exception as e:
            print(f"❌ Erro ao inicializar agentes: {e}")
            raise

        return agents

    async def _rewrite_query_with_history(self, question: str) -> str:
        """Se houver histórico, re-escreve a pergunta para ser autônoma."""
        if not self.conversation_history:
            return question

        print("   [Manager] Reescrevendo a pergunta com base no histórico...")

        # Formata o histórico para o prompt
        formatted_history = "\n".join(
            [f"{msg['role']}: {msg['content']}" for msg in self.conversation_history]
        )

        rewrite_chain = REWRITE_QUERY_PROMPT | self.llm | StrOutputParser()

        try:
            rewritten_question = await rewrite_chain.ainvoke(
                {"chat_history": formatted_history, "question": question}
            )
            print(f"   [Manager] Pergunta reescrita: '{rewritten_question}'")
            return rewritten_question
        except Exception as e:
            print(
                f"   ⚠️ [Manager] Erro ao reescrever pergunta: {e}. Usando a pergunta original."
            )
            return question

    async def process_query(self, query: str) -> dict[str, Any]:
        """
        Processa uma consulta usando o sistema multiagente

        Args:
            query: Pergunta do usuário

        Returns:
            Dict com resposta completa e metadados
        """
        # ✅ v0.4: Inicia o logging da interação
        interaction_id = str(uuid.uuid4())
        start_time = time.time()
        log_data: dict[str, Any] = {
            "interaction_id": interaction_id,
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "original_query": query,
        }

        print(f"🔍 Processando consulta: '{query}'")

        try:
            # 1. ✅ v0.3.1: Reescreve a pergunta ANTES do roteamento
            standalone_query = await self._rewrite_query_with_history(query)

            # 2. Roteamento - decide quais agentes usar com base na pergunta completa
            log_data["standalone_query"] = standalone_query
            routing_decision = self.router.get_routing_decision(standalone_query)
            print(f"   🎯 Roteamento: {routing_decision['selected_agents']}")
            print(f"   📊 Scores: {routing_decision['domain_scores']}")

            # ✅ AJUSTE: Tratamento cordial para perguntas fora de contexto
            # Se o roteador identificar que a pergunta está fora do escopo,
            # retorna uma resposta amigável e encerra o processamento.
            if routing_decision["selected_agents"] == ["out_of_context"]:
                print(
                    "   ⚠️ Pergunta fora de contexto detectada. Acionando busca na web..."
                )
                final_response = await self._handle_web_search(standalone_query, query)
                log_data.update(final_response)
                return final_response

            # 3. Executa os agentes selecionados
            agent_responses = await self._execute_agents(
                standalone_query, routing_decision["selected_agents"]
            )

            # 4. Combina resultados
            final_response = self._combine_responses(
                query, agent_responses, routing_decision
            )

            # ✅ v0.3: Atualiza o histórico da conversa com a nova interação
            self.conversation_history.append({"role": "user", "content": query})
            self.conversation_history.append(
                {"role": "assistant", "content": final_response["final_answer"]}
            )

            print(
                f"   ✅ Consulta processada - {len(agent_responses)} agente(s) responderam"  # noqa: E501
            )
            log_data.update(final_response)
            return final_response  # noqa: TRY300

        except Exception as e:  # noqa: BLE001
            print(f"❌ Erro no processamento: {e}")
            error_response = self._create_error_response(query, str(e))
            log_data.update(error_response)
            return error_response
        finally:
            # ✅ v0.4: Finaliza e registra o log da interação
            end_time = time.time()
            log_data["duration_seconds"] = round(end_time - start_time, 2)
            log_interaction(log_data)

    def clear_history(self) -> None:
        """Limpa o histórico da conversa."""
        self.conversation_history = []
        print("   🧹 Histórico da conversa limpo.")

    async def _execute_agents(
        self, query: str, selected_agents: list[str]
    ) -> list[dict[str, Any]]:
        """Executa os agentes selecionados em paralelo"""
        tasks = []

        for domain in selected_agents:
            agent = self.agents.get(domain)
            if agent:
                # ✅ v0.3: Passa o histórico da conversa para o agente
                # Executa cada agente
                task = (
                    asyncio.create_task(  # ✅ CORREÇÃO: Usa self.conversation_history
                        self._run_agent_safe(agent, query, self.conversation_history)
                    )
                )
                tasks.append(task)

        # Aguarda todas as respostas
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Filtra respostas válidas
        valid_responses: list[dict[str, Any]] = []
        for response in responses:
            if isinstance(response, dict) and response.get("status") == "success":
                valid_responses.append(response)
            elif isinstance(response, Exception):
                print(f"   ⚠️ Agente falhou: {response}")

        return valid_responses

    async def _run_agent_safe(
        self,
        agent: Any,
        query: str,
        conversation_history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Executa um agente com tratamento de erro"""
        try:
            # ✅ v0.3: Passa a query e o histórico para o agente.
            return await agent.invoke(
                {
                    "query": query,  # Esta é a standalone_query
                    "original_query": conversation_history[-1]["content"]
                    if conversation_history
                    else query,  # Passa a última pergunta do usuário
                    "conversation_history": conversation_history,
                }
            )
        except Exception as e:  # noqa: BLE001
            return {
                "agent": agent.name,
                "agent_domain": agent.domain,  # ✅ CORREÇÃO: Usa o atributo correto do objeto agente.
                "answer": f"Erro no agente: {e!s}",
                "sources": [],
                "confidence": 0.0,
                "status": "error",
            }

    async def _handle_web_search(
        self, standalone_query: str, original_query: str
    ) -> dict[str, Any]:
        """Lida com a lógica de fallback de busca na web."""
        web_results = self.web_search_tool.search(standalone_query)

        if not web_results:
            return {
                "query": original_query,
                "final_answer": "Não foram encontrados documentos relevantes para responder a esta pergunta, nem na base interna nem na web.",
                "sources": [],
                "primary_agent": "web_search_fallback",
                "confidence": 0.1,
                "status": "no_documents",
            }

        web_context = "\n\n---\n\n".join(
            [f"Fonte: {res['url']}\nConteúdo: {res['content']}" for res in web_results]
        )
        web_chain = self.web_answer_prompt | self.llm | StrOutputParser()
        answer = await web_chain.ainvoke(
            {"context": web_context, "question": original_query}
        )
        sources = [res["url"] for res in web_results]

        return {
            "query": original_query,
            "final_answer": answer,
            "sources": sources,
            "primary_agent": "web_search_fallback",
            "agent_domain": "general",
            "confidence": 0.60,
            "all_responses": [],
            "status": "success_web_fallback",
        }

    def _combine_responses(
        self,
        query: str,
        agent_responses: list[dict[str, Any]],
        routing_decision: dict[str, Any],
    ) -> dict[str, Any]:
        """Combina as respostas dos agentes em uma resposta final"""

        if not agent_responses:
            return self._create_no_answer_response(query)

        if len(agent_responses) == 1:
            primary_response = agent_responses[0]
            return {
                "query": query,
                "final_answer": primary_response["answer"],
                "sources": primary_response["sources"],
                "primary_agent": primary_response["agent"],
                "agent_domain": primary_response["agent_domain"],
                "confidence": primary_response["confidence"],
                "routing_decision": routing_decision,
                "all_responses": agent_responses,
                "status": "success",
            }

        # Lógica para múltiplos agentes
        combined_answer = self._merge_multiple_answers(agent_responses)
        all_sources = [
            source for resp in agent_responses for source in resp.get("sources", [])
        ]

        return {
            "query": query,
            "final_answer": combined_answer,
            "sources": all_sources,
            "primary_agent": routing_decision["primary_agent"],
            "agent_domains": [resp["agent_domain"] for resp in agent_responses],
            "confidence": max(resp["confidence"] for resp in agent_responses),
            "routing_decision": routing_decision,
            "all_responses": agent_responses,
            "status": "success",
        }

    def _merge_multiple_answers(self, responses: list[dict[str, Any]]) -> str:
        """Combina respostas de múltiplos agentes"""
        if len(responses) == 1:
            return responses[0]["answer"]

        # Para múltiplas respostas, cria uma resposta integrada
        answer_parts: list[str] = ["🔍 **Análise Multiagente - Resposta Integrada**\n"]

        for response in responses:
            answer_parts.append(f"**🏛️ {response['agent'].replace('_', ' ').title()}:**")
            answer_parts.append(response["answer"])
            answer_parts.append("---")

        answer_parts.append(
            "\n**💡 Conclusão Integrada:** Esta análise combina perspectivas de múltiplas fontes jurídicas para uma visão completa."  # noqa: E501
        )

        return "\n".join(answer_parts)

    def _create_no_answer_response(self, query: str) -> dict[str, Any]:
        """Cria resposta quando nenhum agente consegue responder"""
        return {
            "query": query,
            "final_answer": "Não foi possível encontrar uma resposta adequada nas bases jurídicas especializadas.",  # noqa: E501
            "sources": [],
            "primary_agent": None,
            "confidence": 0.0,
            "routing_decision": None,
            "all_responses": [],
            "status": "no_answer",
        }

    def _create_error_response(self, query: str, error: str) -> dict[str, Any]:
        """Cria resposta de erro"""
        return {
            "query": query,
            "final_answer": f"Erro no sistema: {error}",
            "sources": [],
            "primary_agent": None,
            "confidence": 0.0,
            "routing_decision": None,
            "all_responses": [],
            "status": "error",
        }

    def get_agent_info(self) -> dict[str, Any]:
        """Retorna informações sobre os agentes disponíveis"""
        return {
            "total_agents": len(self.agents),
            "available_agents": list(self.agents.keys()),
            "router_ready": hasattr(self, "router"),
            "llm_model": self.llm.model_name,
        }

    # Método síncrono para facilitar o uso
    def process_query_sync(self, query: str) -> dict[str, Any]:
        """Versão síncrona do process_query para facilitar o uso"""
        return asyncio.run(self.process_query(query))
