#!/usr/bin/env python3
"""
Agente roteador para direcionar consultas aos agentes especializados
"""

from typing import Any

from langchain_openai import ChatOpenAI


class LegalRouterAgent:
    """Roteador inteligente para direcionar consultas jurídicas"""

    def __init__(self, llm: ChatOpenAI) -> None:
        self.llm = llm
        self.domains = {
            "constitutional_law": {
                "description": "Direito Constitucional - Constituição Federal, direitos fundamentais",  # noqa: E501
                "keywords": [
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
                "keywords": [
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
                "keywords": [
                    "direitos humanos",
                    "convenção americana",
                    "cadh",
                    "corte interamericana",
                    "tratado internacional",
                ],
            },
        }

    def get_routing_decision(self, query: str) -> dict[str, Any]:
        """
        Decide quais agentes devem processar a consulta

        Args:
            query: A pergunta do usuário

        Returns:
            Dict com decisão de roteamento
        """
        query_lower = query.lower()

        # Calcula scores para cada domínio
        domain_scores = {}
        for domain, info in self.domains.items():
            score = 0
            for keyword in info["keywords"]:
                if keyword in query_lower:
                    score += 1

            # Bonus para matches exatos
            if any(keyword == query_lower for keyword in info["keywords"]):
                score += 2

            domain_scores[domain] = score

        # Seleciona domínios com score > 0
        selected_agents = [
            domain for domain, score in domain_scores.items() if score > 0
        ]

        # Se nenhum domínio específico, usa todos (consulta geral)
        if not selected_agents:
            selected_agents = list(self.domains.keys())
            # Scores baixos para todos
            domain_scores = dict.fromkeys(selected_agents, 0.1)

        # Determina agente principal (maior score)
        primary_agent = (
            max(domain_scores.items(), key=lambda x: x[1])[0]
            if selected_agents
            else None
        )

        return {
            "selected_agents": selected_agents,
            "domain_scores": domain_scores,
            "primary_agent": primary_agent,
            "query": query,
        }
