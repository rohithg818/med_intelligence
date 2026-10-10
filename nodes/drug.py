"""
nodes/drug.py — Drug Agent v2
"""
import time
from state import MedicalState
from retrievers.openfda import retrieve_openfda
from retrievers.rxnorm import retrieve_rxnorm


def drug_node(state: MedicalState) -> MedicalState:
    terms   = state.get("extracted_terms", []) or []
    sources = list(state.get("sources", []) or [])
    log     = state.get("pipeline_log", [])
    results = []
    t0      = time.time()

    for term in terms[:2]:
        fda_data = retrieve_openfda(term)
        if "no fda data" not in fda_data.lower() and "error" not in fda_data.lower():
            results.append(f"[OpenFDA]\n{fda_data}")
            if not any("OpenFDA" in s for s in sources):
                sources.append("OpenFDA (FDA) — https://open.fda.gov")

        rx_data = retrieve_rxnorm(term)
        if "error" not in rx_data.lower() and "no rxnorm" not in rx_data.lower():
            results.append(f"[RxNorm]\n{rx_data}")
            if not any("RxNorm" in s for s in sources):
                sources.append("RxNorm (NLM) — https://rxnav.nlm.nih.gov")

    drug_data = "\n\n---\n\n".join(results) if results else "No drug data retrieved."
    elapsed = round(time.time() - t0, 2)

    log.append({
        "step":    "Drug Agent",
        "apis":    ["OpenFDA", "RxNorm"],
        "hits":    len(results),
        "latency": f"{elapsed}s",
        "sources": sources,
    })

    return {**state, "drug_data": drug_data, "sources": sources, "pipeline_log": log}
