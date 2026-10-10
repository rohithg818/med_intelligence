# Medical Agentic RAG v2
### Intelligent Multi-Agent Pipeline for Grounded Medical Q&A


> authors-  Rohith G  · Sheik Mohammed Hashim 
>

---

## What's New in v2 (vs v1)

| # | Improvement | Type |
|---|---|---|
| 1 | **Quantitative evaluation** — 30-query test set, ROUGE-1/2/L + validator accuracy | Engineered |
| 2 | **Conversation memory** — MedicalState carries history, last 3 turns injected into synthesis | LLM-assisted |
| 3 | **Confidence scoring** — validator returns 0–1 grounded score instead of binary bool | LLM-assisted |
| 4 | **Medical Synonym Trie** — zero-LLM query rewriting using a hand-built trie | **Engineered** |
| 5 | **PubMed fallback** — second retrieval fallback after MedlinePlus, before Wikipedia | Engineered |
| 6 | **Structured JSON router** — single LLM call returning `{intent, terms}` | LLM-assisted |
| 7 | **Streamlit UI v2** — sidebar pipeline trace, tabbed results, confidence bar, history panel | Engineered |

---

## Engineered Component (Non-LLM)

The **Medical Synonym Trie** (`nodes/router.py : MedicalSynonymTrie`) is a hand-crafted
data structure that normalises colloquial medical language into canonical medical terms
**with zero LLM cost**.

```
"my sugar levels are high" → "hyperglycemia"
"hba1c test"               → "glycated hemoglobin test"
"high bp"                  → "hypertension"
"water pill"               → "furosemide"
```

It uses a **longest-match prefix trie** scan over the tokenised query.  
Built once at import time (O(V) where V = vocabulary size), lookups are O(L) in phrase length.  
This improves API retrieval hit rates and costs nothing in tokens or latency.

---

## Architecture

```
User Query
    │
    ▼
[Router Node]          ← Mistral (MISTRAL_MODEL): single JSON call → {intent, terms}
    │
    ▼
[Query Rewriter]       ← ENGINEERED Trie: normalise colloquial terms (no LLM)
    │ (conditional routing by intent)
 ┌──┴────────────────────────────────────┐
 │ diagnosis │  drug   │   lab  │ general │
 │ MedlinePlus│OpenFDA │MedlinePlus│  —   │
 │ PubMed FB  │RxNorm  │PubMed FB │       │
 │ Wiki FB    │        │Wiki FB   │       │
 └──────────────────────────────────────┘
    │ (all merge)
    ▼
[Synthesis Node]       ← Mistral (MISTRAL_MODEL): RAG generate with conversation history
    │
    ▼
[Validator Node ★]     ← Mistral (MISTRAL_MODEL): LLM-as-judge, returns {verdict, score 0–1, reason}
    │
    ▼
Final Answer + Confidence Score + Validation Badge + Sources
```

---

## Setup

```bash
# 1. Clone / unzip the project
cd medical_agentic_rag_v2

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your Mistral API key (and pick a model your tier allows)
echo "MISTRAL_API_KEY=your_key_here" > .env
echo "MISTRAL_MODEL=mistral-tiny" >> .env
#   Free-tier keys are rate-limited on mistral-small-latest (HTTP 429);
#   mistral-tiny is the reliable default. Override via MISTRAL_MODEL.

# 4. Run the Streamlit app
streamlit run streamlit_app.py

# 5. (Optional) Run quantitative evaluation
python evaluation/test_set.py
```

---

## File Structure

```
medical_agentic_rag_v2/
├── state.py                    # Shared pipeline state (MedicalState TypedDict)
├── graph.py                    # LangGraph StateGraph with all nodes + edges
├── streamlit_app.py            # Streamlit UI (sidebar trace, tabs, history)
├── requirements.txt
├── .env                        # MISTRAL_API_KEY
│
├── nodes/
│   ├── router.py               # Router + MedicalSynonymTrie (ENGINEERED)
│   ├── diagnosis.py            # Diagnosis agent (MedlinePlus→PubMed→Wikipedia)
│   ├── drug.py                 # Drug agent (OpenFDA + RxNorm)
│   ├── lab.py                  # Lab agent (MedlinePlus→PubMed→Wikipedia)
│   ├── synthesis.py            # RAG synthesis with conversation memory
│   └── validator.py            # LLM-as-judge with 0–1 confidence score
│
├── retrievers/
│   ├── medlineplus.py          # MedlinePlus NIH API (XML parsing)
│   ├── openfda.py              # OpenFDA drug labels + adverse events
│   ├── rxnorm.py               # RxNorm NLM drug IDs + interactions
│   ├── pubmed.py               # PubMed NCBI E-utilities (NEW)
│   └── wikipedia_api.py        # Wikipedia fallback (MediaWiki API, UA-header)
│
└── evaluation/
    └── test_set.py             # 30-query test set + ROUGE + validator accuracy
```

---

## Evaluation Metrics

Run `python evaluation/test_set.py` to produce:

| Metric | Description |
|---|---|
| Intent Accuracy | % queries correctly routed to diagnosis/drug/lab/general |
| Validator Accuracy | % answers correctly flagged VALID/INVALID vs manual labels |
| Avg. Confidence Score | Mean grounding score (0–1) across all validated queries |
| ROUGE-1 F1 | Unigram overlap between answer and reference |
| ROUGE-2 F1 | Bigram overlap between answer and reference |
| ROUGE-L F1 | Longest common subsequence overlap |

Results saved to `evaluation/results.json` and `evaluation/summary_report.txt`.

---

## API Sources

| API | Used for | Auth |
|---|---|---|
| MedlinePlus (NIH) | Diseases, symptoms, lab tests | None |
| OpenFDA | Drug labels, dosage, adverse events | None |
| RxNorm (NLM) | Drug IDs, interactions | None |
| PubMed (NCBI) | Peer-reviewed abstracts (fallback) | None |
| Wikipedia | General fallback | None |

All APIs are **free with no authentication required**.
