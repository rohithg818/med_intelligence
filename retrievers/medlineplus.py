import requests
import xml.etree.ElementTree as ET


def retrieve_medlineplus(term: str) -> str:
    """
    Retrieve medical information from MedlinePlus (NIH).
    Free API — no key required.
    Covers: diseases, symptoms, conditions, lab tests.
    Docs: https://wsearch.nlm.nih.gov/ws/query
    """
    url = "https://wsearch.nlm.nih.gov/ws/query"
    params = {
        "db":   "healthTopics",
        "term": term,
        "retmax": 3
    }

    try:
        res = requests.get(url, params=params, timeout=10)
        res.raise_for_status()

        # MedlinePlus returns XML — parse it
        root = ET.fromstring(res.text)
        results = []

        for doc in root.findall(".//document"):
            title   = doc.findtext("content[@name='title']", default="")
            snippet = doc.findtext("content[@name='FullSummary']", default="")
            if not snippet:
                snippet = doc.findtext("content[@name='snippet']", default="")

            if title or snippet:
                # clean HTML tags from snippet
                import re
                snippet = re.sub(r"<[^>]+>", "", snippet)
                results.append(f"Topic: {title}\nSummary: {snippet[:600]}")

        if results:
            return "\n\n".join(results)
        return f"No MedlinePlus results found for: {term}"

    except requests.exceptions.Timeout:
        return "MedlinePlus API timeout — no data retrieved."
    except Exception as e:
        return f"MedlinePlus error: {str(e)}"
