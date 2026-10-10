"""
state.py — Shared pipeline state for Medical Agentic RAG v2
Additions over v1:
  - history        : last N conversation turns for multi-turn memory
  - pipeline_log   : step-by-step trace for Streamlit sidebar
  - rewritten_query: output of the new query-rewriting node
  - confidence_score: graded 0–1 validator output (replaces binary bool)
"""
from typing import TypedDict, Optional, List, Dict, Any


class ConversationTurn(TypedDict):
    role:    str   # "user" | "assistant"
    content: str


class MedicalState(TypedDict):
    # ── Input ──────────────────────────────────────────────────────
    user_query:       str
    history:          List[ConversationTurn]   # multi-turn memory

    # ── Router outputs ─────────────────────────────────────────────
    intent:           Optional[str]            # diagnosis|drug|lab|general
    extracted_terms:  Optional[List[str]]      # raw terms from user query
    rewritten_query:  Optional[str]            # ← NEW: normalised query

    # ── Retrieved data ─────────────────────────────────────────────
    diagnosis_data:   Optional[str]
    drug_data:        Optional[str]
    lab_data:         Optional[str]
    knowledge_data:   Optional[str]

    # ── Output ─────────────────────────────────────────────────────
    final_answer:     Optional[str]
    validated:        Optional[bool]
    validation_note:  Optional[str]
    confidence_score: Optional[float]          # ← NEW: 0.0–1.0 grounding score
    sources:          Optional[List[str]]

    # ── Observability ──────────────────────────────────────────────
    pipeline_log:     List[Dict[str, Any]]     # ← NEW: sidebar trace
