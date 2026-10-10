import requests


def retrieve_openfda(term: str) -> str:
    """
    Retrieve drug information from OpenFDA.
    Free API — no key required (rate limit: 1000 req/day without key).
    Covers: drug labels, indications, dosage, warnings, adverse events.
    Docs: https://open.fda.gov/apis/drug/label/
    """
    results = []

    # ── 1. Drug label (indications + dosage + warnings) ──
    label_url = "https://api.fda.gov/drug/label.json"
    label_params = {
        "search": f'(openfda.brand_name:"{term}"+openfda.generic_name:"{term}")',
        "limit":  1
    }

    try:
        res = requests.get(label_url, params=label_params, timeout=10)
        if res.status_code == 200:
            data = res.json().get("results", [{}])[0]

            indications = data.get("indications_and_usage", ["N/A"])[0][:500]
            dosage      = data.get("dosage_and_administration", ["N/A"])[0][:400]
            warnings    = data.get("warnings", ["N/A"])[0][:400]
            brand       = data.get("openfda", {}).get("brand_name", ["Unknown"])[0]
            generic     = data.get("openfda", {}).get("generic_name", ["Unknown"])[0]

            results.append(
                f"Drug: {brand} ({generic})\n"
                f"Indications: {indications}\n"
                f"Dosage: {dosage}\n"
                f"Warnings: {warnings}"
            )
        else:
            results.append(f"No FDA label data found for: {term}")

    except requests.exceptions.Timeout:
        results.append("OpenFDA label API timeout.")
    except Exception as e:
        results.append(f"OpenFDA label error: {str(e)}")

    # ── 2. Adverse events (most reported side effects) ──
    event_url = "https://api.fda.gov/drug/event.json"
    event_params = {
        "search": f'patient.drug.medicinalproduct:"{term}"',
        "count":  "patient.reaction.reactionmeddrapt.exact",
        "limit":  5
    }

    try:
        res = requests.get(event_url, params=event_params, timeout=10)
        if res.status_code == 200:
            events = res.json().get("results", [])
            if events:
                side_effects = [e["term"] for e in events[:5]]
                results.append(f"Most reported side effects: {', '.join(side_effects)}")

    except Exception:
        pass  # adverse events are supplementary — fail silently

    return "\n\n".join(results) if results else f"No FDA data found for: {term}"
