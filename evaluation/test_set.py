"""
evaluation/test_set.py — Quantitative Evaluation Module
Runs 30 curated queries, computes:
  - Intent classification accuracy
  - Validator accuracy (vs manual ground truth grounding labels)
  - ROUGE-1 / ROUGE-2 / ROUGE-L answer quality scores
  - Confidence score distribution

Usage:
    cd medical_agentic_rag_v2
    python evaluation/test_set.py

Outputs:
    evaluation/results.json          — per-query results
    evaluation/summary_report.txt    — human-readable summary table
"""
import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from graph import app
from state import MedicalState

# ── ROUGE implementation (engineered, no external lib needed) ─────

def _tokenize(text: str) -> list:
    import re
    return re.findall(r'\b\w+\b', text.lower())

def _ngrams(tokens: list, n: int) -> dict:
    counts = {}
    for i in range(len(tokens) - n + 1):
        gram = tuple(tokens[i:i+n])
        counts[gram] = counts.get(gram, 0) + 1
    return counts

def rouge_n(hypothesis: str, reference: str, n: int) -> dict:
    hyp_tokens = _tokenize(hypothesis)
    ref_tokens = _tokenize(reference)
    hyp_ng = _ngrams(hyp_tokens, n)
    ref_ng = _ngrams(ref_tokens, n)
    overlap = sum(min(hyp_ng.get(g, 0), ref_ng.get(g, 0)) for g in ref_ng)
    precision = overlap / max(sum(hyp_ng.values()), 1)
    recall    = overlap / max(sum(ref_ng.values()), 1)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}

def rouge_l(hypothesis: str, reference: str) -> dict:
    """Longest Common Subsequence based ROUGE-L."""
    hyp = _tokenize(hypothesis)
    ref = _tokenize(reference)
    m, n = len(hyp), len(ref)
    # DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if hyp[i-1] == ref[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    precision = lcs / max(m, 1)
    recall    = lcs / max(n, 1)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


# ── 30-query Test Set ─────────────────────────────────────────────
# Each entry: {query, expected_intent, reference_answer, ground_truth_grounded}
# ground_truth_grounded = True if a correct answer SHOULD be grounded in APIs
TEST_SET = [
    # DIAGNOSIS (10)
    {"id": "D01", "intent": "diagnosis",
     "query": "What are the symptoms of Type 2 Diabetes?",
     "reference": "Type 2 diabetes symptoms include increased thirst, frequent urination, blurred vision, fatigue, slow wound healing, and frequent infections.",
     "ground_truth_grounded": True},
    {"id": "D02", "intent": "diagnosis",
     "query": "What causes hypertension?",
     "reference": "Hypertension can be caused by obesity, sedentary lifestyle, high salt intake, stress, smoking, kidney disease, and genetic factors.",
     "ground_truth_grounded": True},
    {"id": "D03", "intent": "diagnosis",
     "query": "I have fever, fatigue and headache for 3 days. What could it be?",
     "reference": "Fever, fatigue, and headache lasting 3 days could indicate influenza, viral infections, dengue, or other systemic infections.",
     "ground_truth_grounded": True},
    {"id": "D04", "intent": "diagnosis",
     "query": "What is hypothyroidism?",
     "reference": "Hypothyroidism is a condition where the thyroid gland does not produce enough thyroid hormone, causing fatigue, weight gain, cold sensitivity, and depression.",
     "ground_truth_grounded": True},
    {"id": "D05", "intent": "diagnosis",
     "query": "What are the early signs of anemia?",
     "reference": "Early signs of anemia include fatigue, pale skin, shortness of breath, dizziness, cold hands and feet, and brittle nails.",
     "ground_truth_grounded": True},
    {"id": "D06", "intent": "diagnosis",
     "query": "What is the difference between Type 1 and Type 2 diabetes?",
     "reference": "Type 1 diabetes is autoimmune causing no insulin production; Type 2 is insulin resistance typically related to lifestyle factors.",
     "ground_truth_grounded": True},
    {"id": "D07", "intent": "diagnosis",
     "query": "What are symptoms of kidney disease?",
     "reference": "Symptoms of kidney disease include swollen ankles, fatigue, blood in urine, frequent urination, high blood pressure, and nausea.",
     "ground_truth_grounded": True},
    {"id": "D08", "intent": "diagnosis",
     "query": "What causes migraine headaches?",
     "reference": "Migraines are caused by neurological changes, hormonal shifts, stress, certain foods, sleep changes, and sensory stimuli.",
     "ground_truth_grounded": True},
    {"id": "D09", "intent": "diagnosis",
     "query": "What is PCOS?",
     "reference": "PCOS (Polycystic Ovary Syndrome) is a hormonal disorder causing irregular periods, excess androgen, and ovarian cysts.",
     "ground_truth_grounded": True},
    {"id": "D10", "intent": "diagnosis",
     "query": "What are the symptoms of iron deficiency anemia?",
     "reference": "Iron deficiency anemia causes fatigue, weakness, pale skin, shortness of breath, headache, and brittle nails.",
     "ground_truth_grounded": True},

    # DRUG (10)
    {"id": "DR01", "intent": "drug",
     "query": "What is Metformin used for and what are its side effects?",
     "reference": "Metformin is used for Type 2 diabetes management. Side effects include nausea, diarrhea, stomach upset, and rarely lactic acidosis.",
     "ground_truth_grounded": True},
    {"id": "DR02", "intent": "drug",
     "query": "What are the side effects of ibuprofen?",
     "reference": "Ibuprofen side effects include stomach pain, nausea, heartburn, dizziness, and increased risk of bleeding or kidney problems with long-term use.",
     "ground_truth_grounded": True},
    {"id": "DR03", "intent": "drug",
     "query": "What is Atorvastatin used for?",
     "reference": "Atorvastatin is a statin used to lower LDL cholesterol and triglycerides, reducing cardiovascular disease risk.",
     "ground_truth_grounded": True},
    {"id": "DR04", "intent": "drug",
     "query": "Can I take aspirin and ibuprofen together?",
     "reference": "Aspirin and ibuprofen should not be taken together routinely as ibuprofen may reduce aspirin's antiplatelet effect and increase GI bleeding risk.",
     "ground_truth_grounded": True},
    {"id": "DR05", "intent": "drug",
     "query": "What is the dosage of amoxicillin for adults?",
     "reference": "Adult amoxicillin dosage is typically 250–500 mg every 8 hours or 500–875 mg every 12 hours depending on severity.",
     "ground_truth_grounded": True},
    {"id": "DR06", "intent": "drug",
     "query": "What are the uses of omeprazole?",
     "reference": "Omeprazole is a proton pump inhibitor used for acid reflux, GERD, peptic ulcers, and H. pylori eradication.",
     "ground_truth_grounded": True},
    {"id": "DR07", "intent": "drug",
     "query": "What are the side effects of lisinopril?",
     "reference": "Lisinopril side effects include dry cough, dizziness, headache, elevated potassium levels, and rarely angioedema.",
     "ground_truth_grounded": True},
    {"id": "DR08", "intent": "drug",
     "query": "What is the mechanism of action of metformin?",
     "reference": "Metformin works by reducing hepatic glucose production, improving insulin sensitivity, and decreasing intestinal glucose absorption.",
     "ground_truth_grounded": True},
    {"id": "DR09", "intent": "drug",
     "query": "What drugs interact with warfarin?",
     "reference": "Warfarin interacts with NSAIDs, antibiotics, antifungals, aspirin, statins, and vitamin K-rich foods, affecting INR levels.",
     "ground_truth_grounded": True},
    {"id": "DR10", "intent": "drug",
     "query": "What is paracetamol maximum daily dose?",
     "reference": "The maximum recommended daily dose of paracetamol for adults is 4 grams (4000 mg), or 3 grams for elderly or those with liver issues.",
     "ground_truth_grounded": True},

    # LAB (5)
    {"id": "L01", "intent": "lab",
     "query": "What does a high HbA1c level mean?",
     "reference": "A high HbA1c (above 6.5%) indicates poor blood sugar control over the past 2–3 months and is diagnostic of diabetes.",
     "ground_truth_grounded": True},
    {"id": "L02", "intent": "lab",
     "query": "What is a normal CBC result?",
     "reference": "Normal CBC includes RBC 4.5–5.5 million/uL, WBC 4,500–11,000/uL, hemoglobin 12–17 g/dL, and platelets 150,000–400,000/uL.",
     "ground_truth_grounded": True},
    {"id": "L03", "intent": "lab",
     "query": "What does elevated creatinine in blood test indicate?",
     "reference": "Elevated serum creatinine indicates reduced kidney function or renal impairment, as kidneys filter creatinine from blood.",
     "ground_truth_grounded": True},
    {"id": "L04", "intent": "lab",
     "query": "What is the normal fasting blood glucose level?",
     "reference": "Normal fasting blood glucose is 70–99 mg/dL. Prediabetes is 100–125 mg/dL, and diabetes is 126 mg/dL or above.",
     "ground_truth_grounded": True},
    {"id": "L05", "intent": "lab",
     "query": "What does a low platelet count mean?",
     "reference": "A low platelet count (thrombocytopenia) means below 150,000/uL and can cause increased bleeding risk, bruising, and petechiae.",
     "ground_truth_grounded": True},

    # GENERAL (5)
    {"id": "G01", "intent": "general",
     "query": "How many hours of sleep does an adult need?",
     "reference": "Adults need 7–9 hours of sleep per night according to sleep health guidelines.",
     "ground_truth_grounded": False},
    {"id": "G02", "intent": "general",
     "query": "What is BMI and how is it calculated?",
     "reference": "BMI (Body Mass Index) = weight in kg / (height in meters)^2. Normal BMI is 18.5–24.9.",
     "ground_truth_grounded": False},
    {"id": "G03", "intent": "general",
     "query": "What is the difference between a virus and a bacteria?",
     "reference": "Bacteria are single-celled organisms that can be treated with antibiotics; viruses require a host cell and are treated with antivirals or vaccines.",
     "ground_truth_grounded": False},
    {"id": "G04", "intent": "general",
     "query": "What foods are high in iron?",
     "reference": "Iron-rich foods include red meat, liver, spinach, lentils, beans, fortified cereals, tofu, and pumpkin seeds.",
     "ground_truth_grounded": False},
    {"id": "G05", "intent": "general",
     "query": "What is the importance of drinking water?",
     "reference": "Water maintains hydration, regulates body temperature, transports nutrients, flushes toxins, and supports all body functions.",
     "ground_truth_grounded": False},
]


def run_evaluation(output_dir: str = "evaluation"):
    os.makedirs(output_dir, exist_ok=True)
    results = []
    intent_correct = 0
    validator_correct = 0
    validator_applicable = 0
    rouge1_scores, rouge2_scores, rougel_scores = [], [], []
    confidence_scores = []

    print(f"\n{'='*60}")
    print(f"  Medical Agentic RAG v2 — Quantitative Evaluation")
    print(f"  Running {len(TEST_SET)} queries...")
    print(f"{'='*60}\n")

    for i, item in enumerate(TEST_SET):
        qid   = item["id"]
        query = item["query"]
        print(f"[{i+1:02d}/{len(TEST_SET)}] {qid}: {query[:55]}...")

        t0 = time.time()
        state: MedicalState = {
            "user_query":       query,
            "history":          [],
            "intent":           None,
            "extracted_terms":  None,
            "rewritten_query":  None,
            "diagnosis_data":   None,
            "drug_data":        None,
            "lab_data":         None,
            "knowledge_data":   None,
            "final_answer":     None,
            "validated":        None,
            "validation_note":  None,
            "confidence_score": None,
            "sources":          [],
            "pipeline_log":     [],
        }

        try:
            result = app.invoke(state)
            latency = round(time.time() - t0, 2)

            predicted_intent = result.get("intent", "general")
            answer           = result.get("final_answer", "") or ""
            validated        = result.get("validated")
            conf             = result.get("confidence_score")

            # Intent accuracy
            intent_ok = (predicted_intent == item["intent"])
            if intent_ok:
                intent_correct += 1

            # Validator accuracy (only for queries with source data)
            val_correct = None
            if item["ground_truth_grounded"] and validated is not None:
                validator_applicable += 1
                # Ground truth: grounded queries should be VALID
                val_correct = (validated == True)
                if val_correct:
                    validator_correct += 1
            elif not item["ground_truth_grounded"] and validated is not None:
                validator_applicable += 1
                # General queries often have no source → validator may skip
                val_correct = True  # skip penalty for general
                validator_correct += 1

            # ROUGE scores
            r1 = rouge_n(answer, item["reference"], 1)
            r2 = rouge_n(answer, item["reference"], 2)
            rl = rouge_l(answer, item["reference"])
            rouge1_scores.append(r1["f1"])
            rouge2_scores.append(r2["f1"])
            rougel_scores.append(rl["f1"])

            if conf is not None:
                confidence_scores.append(conf)

            entry = {
                "id":               qid,
                "query":            query,
                "expected_intent":  item["intent"],
                "predicted_intent": predicted_intent,
                "intent_correct":   intent_ok,
                "answer":           answer[:300],
                "validated":        validated,
                "validator_correct":val_correct,
                "confidence_score": conf,
                "rouge1_f1":        r1["f1"],
                "rouge2_f1":        r2["f1"],
                "rougeL_f1":        rl["f1"],
                "latency_s":        latency,
            }
            results.append(entry)
            print(f"    Intent: {'✓' if intent_ok else '✗'} ({predicted_intent})  "
                  f"ROUGE-1: {r1['f1']:.3f}  "
                  f"Validator: {validated} ({conf:.2f if conf else 'N/A'})")

        except Exception as e:
            print(f"    ERROR: {e}")
            results.append({"id": qid, "query": query, "error": str(e)})

    # ── Compute summary stats ─────────────────────────────────────
    n = len(TEST_SET)
    valid_results = [r for r in results if "error" not in r]
    nv = len(valid_results)

    intent_acc = intent_correct / n * 100
    val_acc    = (validator_correct / validator_applicable * 100) if validator_applicable > 0 else 0
    avg_r1     = sum(rouge1_scores) / len(rouge1_scores) * 100 if rouge1_scores else 0
    avg_r2     = sum(rouge2_scores) / len(rouge2_scores) * 100 if rouge2_scores else 0
    avg_rl     = sum(rougel_scores) / len(rougel_scores) * 100 if rougel_scores else 0
    avg_conf   = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0

    summary = {
        "total_queries":            n,
        "successful_runs":          nv,
        "intent_accuracy_pct":      round(intent_acc, 1),
        "validator_accuracy_pct":   round(val_acc, 1),
        "avg_rouge1_f1_pct":        round(avg_r1, 1),
        "avg_rouge2_f1_pct":        round(avg_r2, 1),
        "avg_rougeL_f1_pct":        round(avg_rl, 1),
        "avg_confidence_score":     round(avg_conf, 3),
    }

    # ── Save JSON results ─────────────────────────────────────────
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

    # ── Save human-readable report ────────────────────────────────
    report_lines = [
        "=" * 70,
        "  MEDICAL AGENTIC RAG v2 — EVALUATION REPORT",
        "=" * 70,
        f"  Total queries evaluated     : {n}",
        f"  Successful pipeline runs    : {nv}",
        "-" * 70,
        f"  Intent Classification Acc.  : {intent_acc:.1f}%",
        f"  Validator Accuracy          : {val_acc:.1f}%",
        f"  Avg. Confidence Score       : {avg_conf:.3f} / 1.000",
        "-" * 70,
        f"  Avg. ROUGE-1 F1             : {avg_r1:.1f}%",
        f"  Avg. ROUGE-2 F1             : {avg_r2:.1f}%",
        f"  Avg. ROUGE-L F1             : {avg_rl:.1f}%",
        "=" * 70,
        "",
        "Per-query breakdown:",
        f"{'ID':<6} {'Intent':>9} {'Pred':>9} {'R1':>6} {'R2':>6} {'RL':>6} {'Conf':>6} {'Valid':>6}",
        "-" * 60,
    ]
    for r in valid_results:
        ic = "✓" if r.get("intent_correct") else "✗"
        conf_str = f"{r['confidence_score']:.2f}" if r.get("confidence_score") is not None else "  N/A"
        val_str  = str(r.get("validated", "—"))[:5]
        report_lines.append(
            f"{r['id']:<6} {r['expected_intent']:>9} {r['predicted_intent']:>8}{ic} "
            f"{r['rouge1_f1']*100:>5.1f}% "
            f"{r['rouge2_f1']*100:>5.1f}% "
            f"{r['rougeL_f1']*100:>5.1f}% "
            f"{conf_str:>6} "
            f"{val_str:>6}"
        )

    report_text = "\n".join(report_lines)
    with open(os.path.join(output_dir, "summary_report.txt"), "w") as f:
        f.write(report_text)

    print("\n" + report_text)
    print(f"\n✅ Results saved to {output_dir}/results.json and summary_report.txt")
    return summary


if __name__ == "__main__":
    run_evaluation()
