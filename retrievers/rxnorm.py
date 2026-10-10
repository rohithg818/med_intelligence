import requests


def retrieve_rxnorm(drug_name: str) -> str:
    """
    Retrieve standardized drug info from RxNorm (NLM).
    Free API — no key required.
    Covers: drug name normalization, RxCUI codes, drug interactions.
    Docs: https://rxnav.nlm.nih.gov/RxNormAPIs.html
    """
    results = []

    # ── 1. Get RxCUI (standard drug identifier) ──────────
    rxcui_url = f"https://rxnav.nlm.nih.gov/REST/rxcui.json"
    try:
        res = requests.get(rxcui_url, params={"name": drug_name}, timeout=10)
        if res.status_code == 200:
            ids = res.json().get("idGroup", {}).get("rxnormId", [])
            if ids:
                rxcui = ids[0]
                results.append(f"RxNorm ID (RxCUI): {rxcui} for drug: {drug_name}")

                # ── 2. Get drug properties using RxCUI ──
                props_url = f"https://rxnav.nlm.nih.gov/REST/rxcui/{rxcui}/properties.json"
                props_res = requests.get(props_url, timeout=10)
                if props_res.status_code == 200:
                    props = props_res.json().get("properties", {})
                    results.append(
                        f"Name: {props.get('name', 'N/A')}\n"
                        f"Synonym: {props.get('synonym', 'N/A')}\n"
                        f"Drug Class: {props.get('tty', 'N/A')}"
                    )

                # ── 3. Drug interaction check ────────────
                interact_url = f"https://rxnav.nlm.nih.gov/REST/interaction/interaction.json"
                interact_res = requests.get(
                    interact_url, params={"rxcui": rxcui}, timeout=10
                )
                if interact_res.status_code == 200:
                    interactions = interact_res.json().get(
                        "interactionTypeGroup", []
                    )
                    if interactions:
                        inter_list = []
                        for group in interactions[:1]:
                            for itype in group.get("interactionType", [])[:2]:
                                for pair in itype.get("interactionPair", [])[:2]:
                                    desc = pair.get("description", "")
                                    if desc:
                                        inter_list.append(f"- {desc[:200]}")
                        if inter_list:
                            results.append(
                                "Known interactions:\n" + "\n".join(inter_list)
                            )
            else:
                results.append(f"RxNorm: No standard ID found for '{drug_name}'.")

    except requests.exceptions.Timeout:
        return "RxNorm API timeout."
    except Exception as e:
        return f"RxNorm error: {str(e)}"

    return "\n\n".join(results) if results else f"No RxNorm data for: {drug_name}"
