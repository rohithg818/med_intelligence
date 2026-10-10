"""
graph.py — LangGraph pipeline v2
New node: query_rewriter (trie-based, between router and retrieval agents)
"""
from langgraph.graph import StateGraph, END
from state import MedicalState
from nodes.router import router_node, query_rewriter_node
from nodes.diagnosis import diagnosis_node
from nodes.drug import drug_node
from nodes.lab import lab_node
from nodes.synthesis import synthesis_node
from nodes.validator import validator_node


def route_by_intent(state: MedicalState) -> str:
    return state.get("intent", "general")


def build_graph() -> StateGraph:
    graph = StateGraph(MedicalState)

    graph.add_node("router",         router_node)
    graph.add_node("query_rewriter", query_rewriter_node)   # NEW
    graph.add_node("diagnosis",      diagnosis_node)
    graph.add_node("drug",           drug_node)
    graph.add_node("lab",            lab_node)
    graph.add_node("synthesis",      synthesis_node)
    graph.add_node("validator",      validator_node)

    # router → rewriter → routing
    graph.set_entry_point("router")
    graph.add_edge("router", "query_rewriter")

    # rewriter → conditional routing
    graph.add_conditional_edges(
        "query_rewriter",
        route_by_intent,
        {
            "diagnosis": "diagnosis",
            "drug":      "drug",
            "lab":       "lab",
            "general":   "synthesis",
        }
    )

    graph.add_edge("diagnosis", "synthesis")
    graph.add_edge("drug",      "synthesis")
    graph.add_edge("lab",       "synthesis")
    graph.add_edge("synthesis", "validator")
    graph.add_edge("validator", END)

    return graph.compile()


app = build_graph()
