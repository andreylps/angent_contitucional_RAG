#!/usr/bin/env python3
"""
Agente Roteador que direciona a query para os agentes especializados.

Este agente agora trata perguntas fora de contexto, retornando uma
mensagem cordial quando a confiança de roteamento é baixa.
"""

import json
from typing import Any

from langchain_openai import ChatOpenAI

OUT_OF_CONTEXT_MESSAGE = (
    "Desculpe, minha especialidade é focada em questões jurídicas sobre "
    "Direito Constitucional, Direito do Consumidor e Direitos Humanos. "
    "A sua pergunta parece estar fora desses domínios. "
    "Você poderia, por favor, fazer uma nova pergunta relacionada a esses tópicos?"
)

from ..utils.logger import logger  # noqa: E402, TID252


class LegalRouterAgent:
    """
    Roteador inteligente que usa um LLM para direcionar semanticamente
    consultas jurídicas e identificar perguntas fora de contexto.
    """

    def __init__(self, llm: ChatOpenAI) -> None:
        self.llm = llm
        self.domains = {
            "constitutional_law": {
                "description": "Direito Constitucional - Constituição Federal, direitos fundamentais",  # noqa: E501
                "keywords": [  # Mantido para referência, mas não usado no roteamento
                    "constituição",
                    "constitucional",
                    "artigo 5",
                    "cf/88",
                    "direitos fundamentais",
                    "garantias",
                ],
            },
            "consumer_law": {
                "description": "Direito do Consumidor - CDC, relações de consumo, fornecedor",  # noqa: E501
                "keywords": [  # Mantido para referência, mas não usado no roteamento
                    "consumidor",
                    "fornecedor",
                    "cdc",
                    "defesa do consumidor",
                    "produto",
                    "serviço",
                    "contrato",
                ],
            },
            "human_rights_law": {
                "description": "Direitos Humanos - Convenção Americana, tratados internacionais",  # noqa: E501
                "keywords": [  # Mantido para referência, mas não usado no roteamento
                    "direitos humanos",
                    "convenção americana",
                    "cadh",
                    "corte interamericana",
                    "tratado internacional",
                ],
            },
        }

    def _build_routing_prompt(self, query: str) -> str:
        """Constrói o prompt para o LLM de roteamento."""
        agent_descriptions = "\n".join(
            [
                f"- `{name}`: {info['description']}"
                for name, info in self.domains.items()
            ]
        )

        return f"""
Você é um roteador inteligente em um sistema de assistentes jurídicos.
Sua tarefa é analisar a pergunta do usuário e decidir qual(is) dos seguintes agentes é(são) o(s) mais adequado(s) para responder.
Se a pergunta não estiver relacionada a nenhum dos domínios, retorne uma lista vazia.

Agentes disponíveis:
{agent_descriptions}

Pergunta do usuário:
"{query}"

Responda APENAS com um objeto JSON contendo duas chaves:
1. "selected_agents": uma lista com os nomes dos agentes selecionados.
2. "scores": um dicionário onde cada chave é um agente selecionado e o valor é a confiança da sua escolha (de 0.0 a 1.0).

Exemplo para uma pergunta sobre garantia de produto:
{{"selected_agents": ["consumer_law"], "scores": {{"consumer_law": 0.95}}}}

Exemplo para uma pergunta sobre a constituição e direitos humanos:
{{"selected_agents": ["constitutional_law", "human_rights_law"], "scores": {{"constitutional_law": 0.9, "human_rights_law": 0.7}}}}

Exemplo para uma pergunta sobre futebol (fora de escopo):
{{"selected_agents": [], "scores": {{}}}}
"""  # noqa: E501

    def get_routing_decision(self, query: str) -> dict[str, Any]:
        """
        Decide quais agentes devem processar a consulta usando um LLM.

        Args:
            query: A pergunta do usuário

        Returns:
            Dicionário com a decisão de roteamento.
        """
        prompt = self._build_routing_prompt(query)
        response = self.llm.invoke(prompt)
        content = response.content

        parsed_content = {}
        selected_agents = ["out_of_context"]
        scores = {}
        primary_agent = "out_of_context"

        try:
            if isinstance(content, str):
                # Caso esperado: LLM retorna uma string JSON
                parsed_content = json.loads(content)
            elif isinstance(content, dict):
                # Caso menos comum, mas possível se o LangChain pré-analisar ou o LLM retornar um dict direto  # noqa: E501
                parsed_content = content
            else:
                # Se o conteúdo for None ou outro tipo inesperado
                logger.error(
                    f"LLM retornou conteúdo não-string/dict: {type(content)} - {content}"  # noqa: E501
                )
                msg = "O conteúdo da resposta do LLM não era uma string ou dicionário."
                raise ValueError(  # noqa: TRY004, TRY301
                    msg
                )

            # Agora, extrai com segurança os agentes selecionados do parsed_content
            if isinstance(parsed_content, dict):
                selected_agents = parsed_content.get("selected_agents", [])
                scores = parsed_content.get("scores", {})
                if not isinstance(selected_agents, list):  # Garante que é uma lista
                    selected_agents = []
            else:  # Se parsed_content foi uma lista diretamente (inesperado para este prompt)  # noqa: E501
                selected_agents = []

            # Se a lista de agentes estiver vazia, é fora de contexto
            if not selected_agents:
                selected_agents = ["out_of_context"]
                primary_agent = "out_of_context"
                scores = {}
            else:
                primary_agent = selected_agents[0]  # O primeiro como principal

        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.error(
                f"Erro ao analisar resposta do LLM: {e}. Conteúdo original: {content}"
            )
            selected_agents = ["out_of_context"]  # Fallback em caso de erro
            scores = {}
            primary_agent = "out_of_context"

        return {
            "selected_agents": selected_agents,
            "domain_scores": scores,  # ✅ CORREÇÃO: Usa os scores gerados pelo LLM
            "primary_agent": primary_agent,
            "query": query,
        }
