"""
nodes/router.py — Router Node v2
Improvements over v1:
  1. Single LLM call returning JSON {intent, terms}  (structured output)
  2. Trie-based medical synonym normaliser (ENGINEERED — no LLM)
  3. Separate query_rewriter_node using the synonym trie
"""
import os
import re
import json
import time
from collections import defaultdict
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

# ══════════════════════════════════════════════════════════════════
# ENGINEERED COMPONENT — Medical Synonym Trie
# This replaces a second LLM call for term normalisation.
# The trie maps colloquial phrases → canonical medical terms,
# enabling better API retrieval without any API cost.
# ══════════════════════════════════════════════════════════════════

class TrieNode:
    def __init__(self):
        self.children: dict = {}
        self.canonical: str = None   # set at leaf → replacement term


class MedicalSynonymTrie:
    """
    Prefix-trie that normalises colloquial medical phrases to
    canonical medical terms used by MedlinePlus / OpenFDA.

    E.g.  "my sugar levels are high"  →  "hyperglycemia"
          "chest pain"                →  "angina"
          "blood pressure medication" →  "antihypertensive"
    """

    # ── Knowledge base of colloquial → canonical mappings ─────────
    SYNONYMS = {
        # Symptoms / conditions
        "high sugar":             "hyperglycemia",
        "high blood sugar":       "hyperglycemia",
        "sugar levels are high":  "hyperglycemia",
        "low sugar":              "hypoglycemia",
        "low blood sugar":        "hypoglycemia",
        "sugar levels are low":   "hypoglycemia",
        "high bp":                "hypertension",
        "high blood pressure":    "hypertension",
        "low bp":                 "hypotension",
        "low blood pressure":     "hypotension",
        "heart attack":           "myocardial infarction",
        "chest pain":             "angina",
        "chest tightness":        "angina",
        "shortness of breath":    "dyspnea",
        "cant breathe":           "dyspnea",
        "difficulty breathing":   "dyspnea",
        "stroke":                 "cerebrovascular accident",
        "brain attack":           "cerebrovascular accident",
        "sugar disease":          "diabetes mellitus",
        "diabetes":               "diabetes mellitus",
        "type 2 diabetes":        "diabetes mellitus type 2",
        "type 1 diabetes":        "diabetes mellitus type 1",
        "joint pain":             "arthralgia",
        "knee pain":              "knee arthralgia",
        "back pain":              "dorsalgia",
        "stomach pain":           "abdominal pain",
        "tummy ache":             "abdominal pain",
        "headache":               "cephalalgia",
        "migraine":               "migraine headache",
        "fever":                  "pyrexia",
        "high temperature":       "pyrexia",
        "cold":                   "upper respiratory infection",
        "flu":                    "influenza",
        "kidney disease":         "chronic kidney disease",
        "kidney failure":         "renal failure",
        "liver disease":          "hepatic disease",
        "thyroid problem":        "thyroid disorder",
        "underactive thyroid":    "hypothyroidism",
        "overactive thyroid":     "hyperthyroidism",
        "cholesterol":            "hyperlipidemia",
        "high cholesterol":       "hypercholesterolemia",
        "anemia":                 "iron deficiency anemia",
        "low hemoglobin":         "anemia",
        "low rbc":                "anemia",
        "skin rash":              "dermatitis",
        "itching":                "pruritus",
        "swollen legs":           "peripheral edema",
        "water retention":        "edema",
        "dizziness":              "vertigo",
        "fainting":               "syncope",

        # Lab tests
        "sugar test":             "blood glucose test",
        "blood sugar test":       "blood glucose test",
        "hba1c":                  "glycated hemoglobin test",
        "hba1c test":             "glycated hemoglobin test",
        "a1c test":               "glycated hemoglobin test",
        "cbc":                    "complete blood count",
        "full blood count":       "complete blood count",
        "lft":                    "liver function test",
        "liver test":             "liver function test",
        "kidney test":            "renal function test",
        "kft":                    "renal function test",
        "lipid profile":          "cholesterol blood test",
        "cholesterol test":       "cholesterol blood test",
        "thyroid test":           "thyroid function test",
        "tsh test":               "thyroid stimulating hormone test",

        # Drug colloquialisms
        "sugar pill":             "metformin",
        "water pill":             "furosemide",
        "blood pressure pill":    "antihypertensive",
        "blood pressure medicine": "antihypertensive",
        "painkiller":             "analgesic",
        "pain relief":            "analgesic",
        "sleeping pill":          "sedative hypnotic",
        "allergy medicine":       "antihistamine",
        "antibiotic":             "antibacterial",
        "bp medication":          "antihypertensive",
        "blood thinner":          "anticoagulant",
        "cholesterol medicine":   "statin",
        "insulin":                "insulin therapy",
    }

    def __init__(self):
        self.root = TrieNode()
        for phrase, canonical in self.SYNONYMS.items():
            self._insert(phrase.lower(), canonical)

    def _insert(self, phrase: str, canonical: str):
        node = self.root
        for token in phrase.split():
            if token not in node.children:
                node.children[token] = TrieNode()
            node = node.children[token]
        node.canonical = canonical

    def normalise(self, text: str) -> tuple[str, list[tuple[str, str]]]:
        """
        Scan `text` for longest-match phrases and replace with
        canonical medical terms.
        Returns (rewritten_text, [(original, canonical), ...])

        Matching is punctuation- and case-insensitive: each raw token is
        reduced to its alphanumeric core before the trie lookup, so
        "chest pain!" and "high blood sugar?" still match.
        Unmatched tokens are emitted verbatim, preserving original casing.
        """
        raw_tokens = text.split()
        # alphanumeric core of each token, e.g. "sugar?" → "sugar"
        keys = [re.sub(r"[^a-z0-9]", "", t.lower()) for t in raw_tokens]

        result_tokens = []
        substitutions = []
        n = len(keys)
        i = 0
        while i < n:
            # try longest match first
            node = self.root
            last_match_end = -1
            last_match_canonical = None
            j = i
            while j < n and keys[j] and keys[j] in node.children:
                node = node.children[keys[j]]
                j += 1
                if node.canonical:
                    last_match_end = j
                    last_match_canonical = node.canonical
            if last_match_canonical:
                original = " ".join(keys[i:last_match_end])
                substitutions.append((original, last_match_canonical))
                result_tokens.append(last_match_canonical)
                i = last_match_end
            else:
                result_tokens.append(raw_tokens[i])
                i += 1
        return " ".join(result_tokens), substitutions


# Singleton trie — built once at import time (fast, O(V) in vocab size)
_trie = MedicalSynonymTrie()


# ══════════════════════════════════════════════════════════════════
# Node 1a — Router (single structured JSON LLM call)
# ══════════════════════════════════════════════════════════════════

def router_node(state: MedicalState) -> MedicalState:
    """
    Classifies intent AND extracts terms in ONE LLM call (JSON output).
    Previously v1 made two separate calls — this halves router latency.
    """
    query    = state["user_query"]
    log      = state.get("pipeline_log", [])
    t0       = time.time()

    prompt = f"""You are a medical query classifier and term extractor.
Analyse the query and respond with a single valid JSON object.

Categories:
  diagnosis  → symptoms, diseases, conditions
  drug       → medicines, dosage, side effects, interactions
  lab        → blood tests, lab values, normal ranges
  general    → any other medical question

Query: "{query}"

Respond ONLY with this exact JSON (no markdown, no explanation):
{{"intent": "<category>", "terms": ["<term1>", "<term2>"]}}

Rules:
- intent must be one of: diagnosis, drug, lab, general
- terms: up to 3 key medical entities extracted from the query
"""

    raw = llm.invoke([HumanMessage(content=prompt)]).content.strip()

    # ── Parse JSON robustly ───────────────────────────
    try:
        # strip potential markdown fences
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        parsed = json.loads(clean)
        intent = parsed.get("intent", "general").lower().strip()
        terms  = [str(t).strip() for t in parsed.get("terms", []) if str(t).strip()]
    except (json.JSONDecodeError, KeyError):
        # fallback: try to salvage intent via regex
        intent_match = re.search(r'"intent"\s*:\s*"(\w+)"', raw)
        intent = intent_match.group(1).lower() if intent_match else "general"
        terms  = []

    valid_intents = {"diagnosis", "drug", "lab", "general"}
    if intent not in valid_intents:
        intent = "general"
    if not terms:
        # simple keyword fallback — extract capitalised or quoted words
        terms = re.findall(r'\b[A-Z][a-zA-Z]{2,}\b', query)[:3] or [query[:30]]

    elapsed = round(time.time() - t0, 2)
    log.append({
        "step":    "Router (structured JSON)",
        "intent":  intent,
        "terms":   terms,
        "latency": f"{elapsed}s",
        "note":    "Single LLM call → JSON {intent, terms}"
    })

    return {
        **state,
        "intent":          intent,
        "extracted_terms": terms,
        "sources":         [],
        "pipeline_log":    log,
    }


# ══════════════════════════════════════════════════════════════════
# Node 1b — Query Rewriter (ENGINEERED — Trie-based, zero LLM cost)
# ══════════════════════════════════════════════════════════════════

def query_rewriter_node(state: MedicalState) -> MedicalState:
    """
    ENGINEERED COMPONENT — No LLM used here.
    Uses a Medical Synonym Trie to normalise colloquial terms
    into canonical medical terminology before API retrieval.

    "my sugar levels are high" → "hyperglycemia"
    "hba1c test"               → "glycated hemoglobin test"

    This improves API hit rates without any LLM cost or latency.
    Also replaces extracted_terms with normalised versions.
    """
    query   = state["user_query"]
    terms   = state.get("extracted_terms", []) or []
    log     = state.get("pipeline_log", [])

    # Normalise the full query
    rewritten, subs = _trie.normalise(query)

    # Normalise each extracted term individually
    normalised_terms = []
    term_subs = []
    for term in terms:
        norm, s = _trie.normalise(term)
        normalised_terms.append(norm)
        term_subs.extend(s)

    all_subs = subs + term_subs

    log.append({
        "step":          "Query Rewriter (Trie — engineered)",
        "original":      query,
        "rewritten":     rewritten,
        "substitutions": [f"{o} → {c}" for o, c in all_subs] or ["(no changes)"],
        "note":          "Trie-based synonym normalisation — zero LLM cost"
    })

    return {
        **state,
        "rewritten_query": rewritten,
        "extracted_terms": normalised_terms if normalised_terms else terms,
        "pipeline_log":    log,
    }
