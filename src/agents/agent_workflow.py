# src/agents/graph_workflow.py
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

class AgentState(TypedDict):
    query: str
    routing_decision: dict
    agent_responses: List[dict]
    final_answer: str
    sources: List[str]