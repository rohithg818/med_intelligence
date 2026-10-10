"""
nodes/lab.py — Lab Agent v2
Retrieval chain: MedlinePlus → PubMed → Wikipedia
"""
import time
from state import MedicalState
from retrievers.medlineplus import retrieve_medlineplus
from retrievers.pubmed import retrieve_pubmed
from retrievers.wikipedia_api import retrieve_wikipedia


def lab_node(state: MedicalState) -> MedicalState:
    terms   = state.get("extracted_terms", []) or []
    sources = list(state.get("sources", []) or [])
    log     = state.get("pipeline_log", [])
    results = []
    t0      = time.time()

    for term in terms[:3]:
        search_term = f"{term} lab test"
        data = retrieve_medlineplus(search_term)
        if "error" not in data.lower() and "no results" not in data.lower():
            results.append(data)
            if not any("MedlinePlus" in s for s in sources):
                sources.append("MedlinePlus (NIH) — https://medlineplus.gov")

    # Fallback 1 — PubMed
    if not results and terms:
        data = retrieve_pubmed(f"{terms[0]} laboratory test reference range")
        if "error" not in data.lower() and "no pubmed" not in data.lower():
            results.append(data)
            sources.append("PubMed (NCBI) — https://pubmed.ncbi.nlm.nih.gov")

    # Fallback 2 — Wikipedia
    if not results and terms:
        wiki_data = retrieve_wikipedia(f"{terms[0]} blood test")
        results.append(wiki_data)
        sources.append("Wikipedia — https://wikipedia.org")

    lab_data = "\n\n---\n\n".join(results) if results else "No lab data retrieved."
    elapsed = round(time.time() - t0, 2)

    log.append({
        "step":    "Lab Agent",
        "apis":    ["MedlinePlus", "PubMed (fallback 1)", "Wikipedia (fallback 2)"],
        "hits":    len(results),
        "latency": f"{elapsed}s",
        "sources": sources,
    })

    return {**state, "lab_data": lab_data, "sources": sources, "pipeline_log": log}
