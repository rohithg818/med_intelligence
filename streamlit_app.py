"""
streamlit_app.py — Medical Agentic RAG v2
UI Improvements:
  - Sidebar: real-time pipeline trace (every node's log)
  - Confidence meter (0–1 grounding score as progress bar)
  - Conversation history panel (multi-turn memory via st.session_state)
  - Query rewriting display (shows trie substitutions)
  - Cleaner layout with tabs for Answer / Pipeline / History
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from graph import app
from state import MedicalState

# ── Page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Medical Agentic RAG v2",
    page_icon="🏥",
    layout="wide"
)

# ── Session state init ─────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []          # ConversationTurn list
if "query_log" not in st.session_state:
    st.session_state.query_log = []        # list of {query, result} for history panel

# ── Sidebar ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 Medical RAG v2")
    st.caption("Agentic RAG · LangGraph · Mistral Small")
    st.divider()

    st.markdown("### ⚙️ Pipeline Trace")
    trace_placeholder = st.empty()

    st.divider()
    st.markdown("### 📜 Query History")
    if st.session_state.query_log:
        for i, entry in enumerate(reversed(st.session_state.query_log[-10:])):
            with st.expander(f"Q{len(st.session_state.query_log)-i}: {entry['query'][:40]}…"):
                st.markdown(f"**Intent:** {entry.get('intent','—')}")
                conf = entry.get('confidence_score')
                if conf is not None:
                    st.markdown(f"**Confidence:** {conf:.0%}")
                st.markdown(f"**Answer (snippet):** {entry.get('answer','')[:150]}…")
    else:
        st.caption("No queries yet. Run one below.")

    st.divider()
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.history   = []
        st.session_state.query_log = []
        st.rerun()

# ── Main area ──────────────────────────────────────────────────────
st.title("🏥 Medical Agentic RAG")
st.caption("Powered by Mistral Small · LangGraph · MedlinePlus · OpenFDA · RxNorm · PubMed")

st.warning(
    "⚠️ General information only — always consult a qualified healthcare professional.",
    icon="⚠️"
)

# ── Example query buttons ──────────────────────────────────────────
st.markdown("**Try an example:**")
c1, c2, c3, c4 = st.columns(4)
example = None
if c1.button("🤒 Symptoms",    use_container_width=True): example = "my sugar levels are high and I feel tired"
if c2.button("💊 Drug info",   use_container_width=True): example = "What is Metformin used for and what are its side effects?"
if c3.button("🧪 Lab test",    use_container_width=True): example = "What does a high HbA1c level mean in a blood test?"
if c4.button("📖 General",     use_container_width=True): example = "What is Type 2 Diabetes?"

# ── Query input ────────────────────────────────────────────────────
query = st.text_area(
    "Enter your medical question:",
    value=example if example else "",
    placeholder="e.g. What are the side effects of ibuprofen?",
    height=100,
    key="query_input"
)

# Show previous conversation context
if st.session_state.history:
    with st.expander(f"💬 Continuing conversation ({len(st.session_state.history)//2} prior turn(s))"):
        for turn in st.session_state.history[-6:]:
            role_icon = "🧑" if turn["role"] == "user" else "🤖"
            st.markdown(f"**{role_icon} {turn['role'].capitalize()}:** {turn['content'][:200]}")

col_run, col_new = st.columns([3, 1])
run     = col_run.button("🔍 Run Query",         type="primary", use_container_width=True)
new_con = col_new.button("🔄 New Conversation",  use_container_width=True)

if new_con:
    st.session_state.history = []
    st.rerun()

# ── Run pipeline ───────────────────────────────────────────────────
if run and query.strip():
    initial_state: MedicalState = {
        "user_query":       query.strip(),
        "history":          list(st.session_state.history),
        "intent":           None,
        "extracted_terms":  None,
        "rewritten_query":  None,
        "diagnosis_data":   None,
        "drug_data":        None,
        "lab_data":         None,
        "knowledge_data":   None,
        "final_answer":     None,
        "validated":        None,
        "validation_note":  None,
        "confidence_score": None,
        "sources":          [],
        "pipeline_log":     [],
    }

    with st.status("Running pipeline…", expanded=True) as status:
        st.write("🔀 Classifying intent & extracting terms…")
        st.write("✏️ Rewriting query (trie normaliser)…")
        st.write("🌐 Retrieving from medical APIs…")
        st.write("🤖 Generating answer (RAG)…")
        st.write("🛡️ Validating answer grounding…")
        try:
            result = app.invoke(initial_state)
            status.update(label="✅ Pipeline complete!", state="complete", expanded=False)
        except Exception as e:
            status.update(label="❌ Pipeline failed", state="error")
            st.error(f"Error: {e}")
            st.stop()

    # ── Update session state ───────────────────────────────────────
    st.session_state.history = result.get("history", [])
    st.session_state.query_log.append({
        "query":            query.strip(),
        "intent":           result.get("intent", "—"),
        "answer":           result.get("final_answer", ""),
        "confidence_score": result.get("confidence_score"),
    })

    # ── Update sidebar trace ───────────────────────────────────────
    pipeline_log = result.get("pipeline_log", [])
    trace_md = ""
    for step in pipeline_log:
        sname = step.get("step", "Step")
        trace_md += f"**{sname}**\n"
        for k, v in step.items():
            if k == "step":
                continue
            if isinstance(v, list):
                v_str = ", ".join(str(x) for x in v[:5])
            else:
                v_str = str(v)[:80]
            trace_md += f"- `{k}`: {v_str}\n"
        trace_md += "\n"
    trace_placeholder.markdown(trace_md if trace_md else "_No trace available_")

    # ── Result display ─────────────────────────────────────────────
    intent = result.get("intent", "—")
    terms  = result.get("extracted_terms") or []
    rw     = result.get("rewritten_query", "")
    conf   = result.get("confidence_score")

    tab1, tab2, tab3 = st.tabs(["📋 Answer", "🔍 Pipeline Details", "📚 Sources"])

    with tab1:
        # Metrics row
        m1, m2, m3 = st.columns(3)
        intent_emoji = {"diagnosis": "🩺", "drug": "💊", "lab": "🧪", "general": "📖"}.get(intent, "❓")
        m1.metric("Intent",          f"{intent_emoji} {intent.capitalize()}")
        m2.metric("Extracted Terms", ", ".join(terms[:3]) if terms else "—")
        if conf is not None:
            m3.metric("Grounding Score", f"{conf:.0%}")
        else:
            m3.metric("Grounding Score", "N/A")

        # Confidence bar
        if conf is not None:
            bar_color = "🟢" if conf >= 0.7 else ("🟡" if conf >= 0.4 else "🔴")
            st.markdown(f"**{bar_color} Grounding Confidence: {conf:.0%}**")
            st.progress(conf)

        # Validation badge
        validated = result.get("validated")
        note      = result.get("validation_note", "")
        if validated is True:
            st.success(f"✅ {note}")
        elif validated is False:
            st.error(f"❌ {note}")
        else:
            st.info("ℹ️ Validation not applicable (general query).")

        # Answer
        st.markdown("### Answer")
        st.markdown(result.get("final_answer", "No answer generated."))

    with tab2:
        st.markdown("### 🔍 Pipeline Trace")

        # Query rewriting
        if rw and rw.lower() != query.strip().lower():
            st.info(f"✏️ **Query rewritten** (Trie normaliser):\n\n"
                    f"**Original:** {query.strip()}\n\n"
                    f"**Rewritten:** {rw}")

        for step in pipeline_log:
            sname = step.get("step", "Step")
            with st.expander(f"🔷 {sname}", expanded=False):
                for k, v in step.items():
                    if k == "step":
                        continue
                    if isinstance(v, list):
                        st.markdown(f"**{k}:**")
                        for item in v:
                            st.markdown(f"  - {item}")
                    else:
                        st.markdown(f"**{k}:** `{v}`")

    with tab3:
        sources = result.get("sources") or []
        if sources:
            st.markdown("### 📚 Sources Used")
            for s in sources:
                st.markdown(f"- {s}")
        else:
            st.info("No external sources used for this query.")

elif run and not query.strip():
    st.warning("Please enter a question first.")
