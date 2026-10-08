from __future__ import annotations

import json

import pandas as pd

NO_DATA_MSG = ("There is no recorded reaction data for this compound in the database, always consult a doctor or pharmacist.")

SEVERITY_NOTE = ("This severity grading is based on the described mechanism and potential clinical harm, ")

PAYLOAD_COLS = ["mechanism", "direction", "exposure_change", "interaction_class",
                "detail", "actor", "affected", "active_metabolites", "efficacy_loss", "description"]

SYSTEM_PROMPT = """You are a clinical pharmacology assistant. You receive structured, already-verified
drug-drug interaction records. Do NOT add interactions that are not in the records.
Return ONLY JSON: {"severity": "High|Moderate|Low", "rationale_en": str, "rationale_ar": str,
"advice_ar": str, "confidence": "high|medium|low"}.
Guidance: grade by the potential clinical harm of the described mechanism (e.g. QT/bleeding/serotonergic
toxicity, large exposure increase of a narrow-therapeutic-index drug => High; loss of efficacy of a
critical drug => at least Moderate). If the records are not enough to judge, use confidence "low".
Keep rationales under 2 sentences, plain language, no dosing instructions."""


def build_payload(drug_a: str, drug_b: str, rows: pd.DataFrame, warnings: dict | None = None) -> dict:
    """rows: all rows of interactions.parquet for the pair (any order)."""
    recs = rows[PAYLOAD_COLS].astype(object).where(rows[PAYLOAD_COLS].notna(), None).to_dict("records")
    return {"drug_a": drug_a, "drug_b": drug_b, "n_records": len(recs),
            "records": recs[:10],                    # cap tokens for small local models
            "market_warnings": warnings or {}}       # e.g. 7 warning_* flags from eg_drugs


def grade_with_ollama(payload: dict, model: str = "llama3.1:8b") -> dict:
    import ollama  # pip install ollama
    resp = ollama.chat(model=model, format="json", options={"temperature": 0},
                       messages=[{"role": "system", "content": SYSTEM_PROMPT},
                                 {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}])
    out = json.loads(resp["message"]["content"])
    if out.get("severity") not in {"High", "Moderate", "Low"}:
        out["severity"], out["confidence"] = None, "low"     # never trust malformed output
    out["disclaimer_ar"] = SEVERITY_NOTE_AR
    return out