#!/usr/bin/env python3
"""
Gerenciador principal do sistema multiagente RAG jurídico
"""

import asyncio
import os
from typing import Any

from dotenv import load_dotenv
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

    async def process_query(
        self, query: str, conversation_history: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        """
        Processa uma consulta usando o sistema multiagente

        Args:
            query: Pergunta do usuário
            conversation_history: Histórico da conversa (opcional)

        Returns:
            Dict com resposta completa e metadados
        """
        print(f"🔍 Processando consulta: '{query}'")

        try:
            # 1. Roteamento - decide quais agentes usar
            routing_decision = self.router.get_routing_decision(query)
            print(f"   🎯 Roteamento: {routing_decision['selected_agents']}")
            print(f"   📊 Scores: {routing_decision['domain_scores']}")

            # ✅ AJUSTE: Tratamento cordial para perguntas fora de contexto
            # Se o roteador identificar que a pergunta está fora do escopo,
            # retorna uma resposta amigável e encerra o processamento.
            if routing_decision["selected_agents"] == ["out_of_context"]:
                print("   ⚠️ Pergunta fora de contexto detectada.")
                friendly_response = (
                    "Olá! Eu sou um assistente jurídico especializado em legislação brasileira, "
                    "com foco na Constituição, Direito do Consumidor e Direitos Humanos.\n\n"
                    "Sua pergunta parece estar fora da minha área de conhecimento. "
                    "Poderia, por favor, fazer uma pergunta dentro desses domínios?"
                )
                return {
                    "query": query,
                    "final_answer": friendly_response,
                    "routing_decision": routing_decision,
                    "status": "out_of_context",
                }

            # 2. Executa os agentes selecionados
            agent_responses = await self._execute_agents(
                query, routing_decision["selected_agents"], conversation_history
            )

            # 3. Combina resultados
            final_response = self._combine_responses(
                query, agent_responses, routing_decision
            )

            print(
                f"   ✅ Consulta processada - {len(agent_responses)} agente(s) responderam"  # noqa: E501
            )
            return final_response  # noqa: TRY300

        except Exception as e:  # noqa: BLE001
            print(f"❌ Erro no processamento: {e}")
            return self._create_error_response(query, str(e))

    async def _execute_agents(
        self,
        query: str,
        selected_agents: list[str],
        conversation_history: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Executa os agentes selecionados em paralelo"""
        tasks = []

        for domain in selected_agents:
            agent = self.agents.get(domain)
            if agent:
                # Executa cada agente
                task = asyncio.create_task(
                    self._run_agent_safe(agent, query, conversation_history)
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
        history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Executa um agente com tratamento de erro"""
        try:
            return agent.invoke(
                query
            )  # ✅ CORREÇÃO: Remove a passagem do histórico, que não é usado pelo agente.
        except Exception as e:  # noqa: BLE001
            return {
                "agent": agent.name,
                "agent_domain": agent.domain,  # ✅ CORREÇÃO: Usa o atributo correto do objeto agente.
                "answer": f"Erro no agente: {e!s}",
                "sources": [],
                "confidence": 0.0,
                "status": "error",
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
    def process_query_sync(
        self, query: str, conversation_history: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        """Versão síncrona do process_query para facilitar o uso"""
        return asyncio.run(self.process_query(query, conversation_history))
