"""
nodes/validator.py — Validator Node v2
Improvement: returns a graded confidence score 0.0–1.0 alongside
VALID/INVALID verdict, replacing the binary boolean output.
This maps to RAGAS faithfulness metric and is more academically rigorous.
"""
import os
import re
import time
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from state import MedicalState

load_dotenv()

llm = ChatMistralAI(
    api_key=os.getenv("MISTRAL_API_KEY"),
    model=os.getenv("MISTRAL_MODEL", "mistral-tiny"),
    temperature=0
)


def validator_node(state: MedicalState) -> MedicalState:
    """
    Node 4 — Validator v2 (Novel Contribution)
    LLM-as-judge hallucination checker with:
      - VALID / INVALID verdict
      - confidence_score: 0.0–1.0 grounding score
      - brief reason for any INVALID verdict
    """
    final_answer = state.get("final_answer", "")
    log          = state.get("pipeline_log", [])
    t0           = time.time()

    # Collect raw source data
    raw_parts = []
    for key in ["diagnosis_data", "drug_data", "lab_data", "knowledge_data"]:
        if state.get(key):
            raw_parts.append(state[key])
    raw_data = "\n\n".join(raw_parts)

    if not raw_data.strip():
        log.append({
            "step":             "Validator",
            "verdict":          "SKIPPED",
            "confidence_score": None,
            "note":             "No source data — cannot validate general answers",
        })
        return {
            **state,
            "validated":        False,
            "validation_note":  "Could not validate — no source data was retrieved.",
            "confidence_score": None,
            "pipeline_log":     log,
        }

    prompt = f"""You are a precise medical fact-checker evaluating answer grounding.

── Raw API Source Data (ground truth) ──
{raw_data[:3000]}

── Generated Answer (to check) ──
{final_answer[:1500]}

Task:
1. Check every factual claim in the Generated Answer against the Source Data.
2. Assign a CONFIDENCE SCORE between 0.0 and 1.0:
   - 1.0 = every claim is directly supported by the source data
   - 0.7–0.9 = mostly supported, minor gaps
   - 0.4–0.6 = partially supported, some claims unverified
   - 0.0–0.3 = major claims contradict or absent from source data
3. Give a VERDICT: VALID (score ≥ 0.6) or INVALID (score < 0.6)

Respond with ONLY this exact JSON (no markdown, no explanation):
{{"verdict": "VALID", "confidence": 0.92, "reason": "All claims supported."}}
or
{{"verdict": "INVALID", "confidence": 0.35, "reason": "Claim about dosage not found in source."}}
"""

    raw_result = llm.invoke([HumanMessage(content=prompt)]).content.strip()

    # ── Parse JSON response ───────────────────────────────────────
    verdict    = "INVALID"
    confidence = 0.5
    reason     = "Could not parse validator response."

    try:
        clean  = re.sub(r"```(?:json)?|```", "", raw_result).strip()
        parsed = re.search(r'\{.*\}', clean, re.DOTALL)
        if parsed:
            obj        = __import__('json').loads(parsed.group())
            verdict    = str(obj.get("verdict", "INVALID")).upper().strip()
            confidence = float(obj.get("confidence", 0.5))
            reason     = str(obj.get("reason", ""))
            confidence = max(0.0, min(1.0, confidence))   # clamp to [0,1]
    except Exception:
        # fallback: scan for VALID/INVALID keyword
        if "VALID" in raw_result.upper():
            verdict    = "VALID"
            confidence = 0.75
            reason     = "Answer appears consistent with source."
        else:
            verdict    = "INVALID"
            confidence = 0.3
            reason     = raw_result[:200]

    validated = (verdict == "VALID")
    elapsed   = round(time.time() - t0, 2)

    log.append({
        "step":             "Validator (LLM-as-judge)",
        "verdict":          verdict,
        "confidence_score": confidence,
        "reason":           reason,
        "latency":          f"{elapsed}s",
        "note":             "Graded 0–1 faithfulness score (v2)",
    })

    note = (
        f"Answer is consistent with retrieved sources. (confidence: {confidence:.0%})"
        if validated else
        f"INVALID — {reason} (confidence: {confidence:.0%})"
    )

    return {
        **state,
        "validated":        validated,
        "validation_note":  note,
        "confidence_score": confidence,
        "pipeline_log":     log,
    }
