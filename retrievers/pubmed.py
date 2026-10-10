"""
retrievers/pubmed.py — PubMed E-utilities retriever
Free, no API key required. Authoritative peer-reviewed source.
Used as second fallback: MedlinePlus → PubMed → Wikipedia
"""
import requests
import xml.etree.ElementTree as ET
import re


def retrieve_pubmed(term: str, max_results: int = 3) -> str:
    """
    Retrieve abstracts from PubMed via NCBI E-utilities.
    Step 1: esearch  — get article IDs for the term
    Step 2: efetch   — get title + abstract for each ID
    Free API, no key needed (rate limit: 3 req/sec unauthenticated).
    """
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    # ── Step 1: Search ────────────────────────────────────────────
    try:
        search_res = requests.get(
            f"{base}/esearch.fcgi",
            params={
                "db":      "pubmed",
                "term":    f"{term}[MeSH Terms] OR {term}[Title/Abstract]",
                "retmax":  max_results,
                "retmode": "json",
                "sort":    "relevance",
            },
            timeout=10
        )
        search_res.raise_for_status()
        id_list = search_res.json().get("esearchresult", {}).get("idlist", [])
    except requests.exceptions.Timeout:
        return "PubMed search timeout."
    except Exception as e:
        return f"PubMed search error: {str(e)}"

    if not id_list:
        return f"No PubMed results for: {term}"

    # ── Step 2: Fetch abstracts ────────────────────────────────────
    try:
        fetch_res = requests.get(
            f"{base}/efetch.fcgi",
            params={
                "db":      "pubmed",
                "id":      ",".join(id_list),
                "rettype": "abstract",
                "retmode": "xml",
            },
            timeout=15
        )
        fetch_res.raise_for_status()
    except requests.exceptions.Timeout:
        return "PubMed fetch timeout."
    except Exception as e:
        return f"PubMed fetch error: {str(e)}"

    # ── Parse XML ─────────────────────────────────────────────────
    try:
        root = ET.fromstring(fetch_res.text)
    except ET.ParseError:
        return "PubMed: could not parse response XML."

    results = []
    for article in root.findall(".//PubmedArticle"):
        title_el    = article.find(".//ArticleTitle")
        abstract_el = article.find(".//AbstractText")
        pmid_el     = article.find(".//PMID")

        title    = (title_el.text or "").strip()    if title_el    is not None else ""
        abstract = (abstract_el.text or "").strip() if abstract_el is not None else ""
        pmid     = (pmid_el.text or "").strip()     if pmid_el     is not None else ""

        # clean XML artifacts
        abstract = re.sub(r"\s+", " ", abstract)

        if title or abstract:
            snippet = abstract[:500] if abstract else "(no abstract)"
            url     = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""
            results.append(
                f"Title: {title}\n"
                f"Abstract: {snippet}\n"
                f"URL: {url}"
            )

    if results:
        return "\n\n---\n\n".join(results)
    return f"PubMed: no abstracts available for '{term}'."
