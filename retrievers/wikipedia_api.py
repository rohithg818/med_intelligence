import re
import requests

API_URL   = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "medical-agentic-rag/2.0 (student research project)"
TIMEOUT   = 15


def _get(params: dict) -> dict:
    """Call the MediaWiki API with a proper User-Agent (Wikimedia 403s without one)."""
    resp = requests.get(
        API_URL,
        params={**params, "format": "json"},
        headers={"User-Agent": USER_AGENT},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def retrieve_wikipedia(term: str) -> str:
    """
    Retrieve general medical knowledge from Wikipedia.
    Free — no key required. Used as fallback when APIs return nothing.

    Uses the MediaWiki API directly instead of the `wikipedia` package,
    which sends no User-Agent and is rejected with HTTP 403 by Wikimedia.
    """
    try:
        # 1. Preferred: the term itself is an article title
        #    (full-text search ranks poorly for single medical terms)
        results = []
        direct = _get({
            "action": "query",
            "prop":  "extracts|info",
            "inprop": "url",
            "exintro":  1,
            "explaintext": 1,
            "exsentences": 5,
            "redirects": 1,
            "titles": term,
        })
        for pdata in direct.get("query", {}).get("pages", {}).values():
            if pdata.get("missing") is None and pdata.get("extract"):
                results.append(pdata)

        # 2. Fallback: full-text search candidates
        if not results:
            search = _get({
                "action": "query",
                "list":   "search",
                "srsearch": term,
                "srlimit":  3,
                "srnamespace": 0,
            })
            hits = [h["title"] for h in search.get("query", {}).get("search", [])]
            if not hits:
                return f"No Wikipedia results for: {term}"
            page = _get({
                "action": "query",
                "prop":  "extracts|info",
                "inprop": "url",
                "exintro":  1,
                "explaintext": 1,
                "exsentences": 5,
                "redirects": 1,
                "titles": "|".join(hits),
            })
            results = sorted(
                page.get("query", {}).get("pages", {}).values(),
                key=lambda p: p.get("index", 99),
            )

        # 3. First usable extract wins
        for pdata in results:
            title   = pdata.get("title", "")
            extract = (pdata.get("extract") or "").strip()
            url     = pdata.get("fullurl", "")
            if not extract:
                continue
            # skip disambiguation pages — try the next candidate
            if re.match(r"^(.*\s)?(may refer to|disambiguation)\b", extract, re.IGNORECASE):
                continue

            summary = re.sub(r"\s+", " ", extract).strip()
            return (
                f"Source: Wikipedia — {title}\n"
                f"URL: {url}\n\n"
                f"{summary}"
            )

        return f"Wikipedia: Could not retrieve a clear article for '{term}'."

    except Exception as e:
        return f"Wikipedia error: {str(e)}"
