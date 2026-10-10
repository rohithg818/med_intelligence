"""
nodes/synthesis.py — Synthesis Node v2
Improvement: injects last N conversation turns (history) into the
prompt, enabling multi-turn medical dialogue.
"""
import os
import time
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from state import MedicalState

load_dotenv()

llm = ChatMistralAI(
    api_key=os.getenv("MISTRAL_API_KEY"),
    model=os.getenv("MISTRAL_MODEL", "mistral-tiny"),
    temperature=0.1,
    max_tokens=1024
)

MAX_HISTORY_TURNS = 3   # pass last 3 turns to synthesis


def synthesis_node(state: MedicalState) -> MedicalState:
    query   = state["user_query"]
    log     = state.get("pipeline_log", [])
    t0      = time.time()

    # ── Assemble retrieved context ────────────────────────────────
    context_parts = []
    if state.get("diagnosis_data"):
        context_parts.append(f"=== DIAGNOSIS / DISEASE INFORMATION ===\n{state['diagnosis_data']}")
    if state.get("drug_data"):
        context_parts.append(f"=== DRUG / MEDICATION INFORMATION ===\n{state['drug_data']}")
    if state.get("lab_data"):
        context_parts.append(f"=== LAB TEST INFORMATION ===\n{state['lab_data']}")
    if state.get("knowledge_data"):
        context_parts.append(f"=== GENERAL MEDICAL KNOWLEDGE ===\n{state['knowledge_data']}")
    context = "\n\n".join(context_parts)

    # ── Build conversation history block ──────────────────────────
    history = state.get("history", []) or []
    recent  = history[-(MAX_HISTORY_TURNS * 2):]   # last N turns × 2 (user+assistant)
    history_block = ""
    if recent:
        lines = []
        for turn in recent:
            role = "Patient" if turn["role"] == "user" else "Assistant"
            lines.append(f"{role}: {turn['content'][:300]}")
        history_block = "── Conversation History (last turns) ──\n" + "\n".join(lines) + "\n\n"

    sources_text = "\n".join(state.get("sources", []) or [])
    rewritten    = state.get("rewritten_query") or query

    prompt = f"""You are a helpful medical information assistant.
Use ONLY the provided context to answer the question.
Do NOT hallucinate or add information not in the context.
Always remind the user to consult a doctor for personal medical advice.

{history_block}── Retrieved Medical Context ──
{context[:3000]}

── Sources ──
{sources_text}

── Current Question ──
{rewritten}

── Instructions ──
1. Answer clearly in plain language a patient can understand.
2. If the history above provides relevant context, refer to it naturally.
3. Structure your answer with short sections if needed.
4. End with: "⚠️ Always consult a qualified doctor before making any medical decisions."
"""

    final_answer = llm.invoke([HumanMessage(content=prompt)]).content.strip()
    elapsed = round(time.time() - t0, 2)

    log.append({
        "step":           "Synthesis (RAG Generate)",
        "history_turns":  len(recent) // 2,
        "context_chars":  len(context),
        "latency":        f"{elapsed}s",
        "note":           f"Injected {len(recent)//2} prior turn(s) into prompt",
    })

    # ── Append this turn to history ───────────────────────────────
    updated_history = list(history) + [
        {"role": "user",      "content": query},
        {"role": "assistant", "content": final_answer},
    ]

    return {
        **state,
        "final_answer": final_answer,
        "history":      updated_history,
        "pipeline_log": log,
    }
