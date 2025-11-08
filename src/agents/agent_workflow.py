# src/agents/graph_workflow.py
from typing import Any, TypedDict


class AgentState(TypedDict):
    """
    Representa o estado do nosso grafo de agentes.
    Este estado é passado entre os nós do grafo.
    """

    query: str
    routing_decision: dict
    agent_responses: list[dict[str, Any]]
    final_answer: str
    sources: list[dict[str, Any]]
    primary_agent: str
    agent_domain: str
    confidence: float
