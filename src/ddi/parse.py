"""
src/ddi/parse.py
================
Parse the free-text `Interaction Description` of db_drug_interactions.csv into
structured, model-ready columns and save them as `interactions.parquet`.

Strategy (template masking)
---------------------------
1. Lower-case the sentence and replace the two drug names with the tokens
   DRUG1 / DRUG2  ->  the ~191k sentences collapse into a few dozen templates.
2. Match the masked sentence against an ordered list of regex rules.
3. Anything not matched is reported (top unmatched templates) so a new rule can
   be added -- nothing is silently dropped or guessed.

Output columns
--------------
drug1, drug2          normalized ingredient names (via src.ingredients.norm_ing)
drug1_raw, drug2_raw  original names
pair_key              order-independent key "a||b" (use for GroupKFold / dedup)
mechanism             pharmacodynamic | serum_concentration | metabolism |
                      absorption | bioavailability | excretion | protein_binding |
                      efficacy | adverse_risk | other_pk
direction             increase | decrease     (what the sentence says, literally)
exposure_change       up | down | NA          (net effect on the AFFECTED drug's
                      level: metabolism/excretion are inverted)
actor / affected      drug1 | drug2 | unspecified
detail                e.g. "photosensitizing", "adverse effects", "hypotension"
active_metabolites    bool
efficacy_loss         bool  ("... resulting in a loss in efficacy")
interaction_class     coarse label derived from mechanism+direction (see classify)
rule, template        which rule matched + the masked sentence (for debugging)

NOTE: the source data has NO severity column. `interaction_class` is a
mechanism-based grouping, NOT clinical severity (High/Moderate/Low).

Usage
-----
python -m src.ddi.parse \
    --ddi data/raw/db_drug_interactions.csv \
    --out data/processed/interactions.parquet \
    --report-dir reports
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

try:  # project normalizer (vocab-aware); fall back to a minimal one
    from src.ingredients import norm_ing
except ImportError:  # pragma: no cover
    def norm_ing(x: str, *_, **__) -> str:
        return re.sub(r"\s+", " ", str(x).strip().lower())

# --------------------------------------------------------------------------- #
# 1. Masking
# --------------------------------------------------------------------------- #
_TOKENS = {"DRUG1": "drug1", "DRUG2": "drug2"}


def mask_sentence(text: str, d1: str, d2: str) -> str:
    """Replace drug names in `text` with DRUG1/DRUG2 (longest name first,
    whole-word only, so 'iron' does not hit 'environment')."""
    t = str(text).lower().strip().rstrip(".")
    names = sorted(
        [(str(d1).lower().strip(), "DRUG1"), (str(d2).lower().strip(), "DRUG2")],
        key=lambda x: -len(x[0]),
    )
    for name, tok in names:
        if not name:
            continue
        t = re.sub(rf"(?<![a-z0-9]){re.escape(name)}(?![a-z0-9])", tok, t)
    return re.sub(r"\s+", " ", t)


# --------------------------------------------------------------------------- #
# 2. Rules  (ordered: first match wins)
# --------------------------------------------------------------------------- #
D = r"(DRUG[12])"

# "X may increase/decrease the <detail> activities of Y"
R_PD = re.compile(rf"^{D} may (increase|decrease) the (.+?) (?:activities|activity) of {D}$")

# "X may increase/decrease the excretion rate of Y [which could result in ...]"
R_EXCR_RATE = re.compile(
    rf"^{D} may (increase|decrease) the (excretion rate|excretion|absorption|"
    rf"serum concentration|metabolism|bioavailability) of {D}(?: which could result in (.+))?$"
)

# "The <pk term> of [the active metabolites of] Y can be increased/decreased/reduced [when ...]"
R_PK = re.compile(
    rf"^the (serum concentration|metabolism|absorption|bioavailability|excretion rate|excretion|"
    rf"protein binding|therapeutic efficacy) of (?:the active metabolites of )?{D} "
    rf"(?:can|may) be (increased|decreased|reduced)(?: when .*| and .*| resulting .*)?$"
)

# "The risk or severity of <detail> can be increased/decreased when X is combined with Y"
R_RISK = re.compile(
    rf"^the risk or severity of (.+?) (?:can|may) be (increased|decreased)(?: when .*)?$"
)

# "The risk of a <detail> to/in Y is increased when ..."   (hypersensitivity etc.)
R_RISK2 = re.compile(rf"^the risk of (?:a |an )?(.+?) (?:to|in|with) {D} is (increased|decreased)(?: when .*)?$")

_MECH = {
    "serum concentration": "serum_concentration",
    "metabolism": "metabolism",
    "absorption": "absorption",
    "bioavailability": "bioavailability",
    "excretion rate": "excretion",
    "excretion": "excretion",
    "protein binding": "protein_binding",
    "therapeutic efficacy": "efficacy",
}


def _dir(word: str) -> str:
    return "increase" if word.startswith("increase") else "decrease"  # reduced/decreased -> decrease


def _other(tok: str) -> str:
    return "DRUG2" if tok == "DRUG1" else "DRUG1"


def _exposure(mech: str, direction: str) -> str | None:
    """Net effect on the AFFECTED drug's exposure."""
    if mech in {"serum_concentration", "bioavailability", "absorption"}:
        return "up" if direction == "increase" else "down"
    if mech in {"metabolism", "excretion"}:  # more clearance -> lower exposure
        return "down" if direction == "increase" else "up"
    return None


def classify(mech: str, direction: str, exposure: str | None) -> str:
    """Coarse, mechanism-based class. NOT clinical severity."""
    if mech == "adverse_risk":
        return "adverse_risk_up" if direction == "increase" else "adverse_risk_down"
    if mech == "pharmacodynamic":
        return "pd_potentiation" if direction == "increase" else "pd_reduction"
    if mech == "efficacy":
        return "efficacy_gain" if direction == "increase" else "efficacy_loss"
    if exposure == "up":
        return "exposure_up"
    if exposure == "down":
        return "exposure_down"
    return "other"


def parse_masked(t: str) -> dict | None:
    """Parse one masked sentence. Returns None if no rule matches."""
    base = {"detail": None, "active_metabolites": "active metabolites" in t,
            "efficacy_loss": ("loss in efficacy" in t) or ("reduction in efficacy" in t)}

    if m := R_PD.match(t):
        actor, dirw, detail, aff = m.groups()
        return {**base, "rule": "pd", "mechanism": "pharmacodynamic", "direction": _dir(dirw),
                "actor": _TOKENS[actor], "affected": _TOKENS[aff], "detail": detail}

    if m := R_EXCR_RATE.match(t):
        actor, dirw, what, aff, detail = m.groups()
        return {**base, "rule": "pk_actor", "mechanism": _MECH.get(what, "other_pk"),
                "direction": _dir(dirw), "actor": _TOKENS[actor], "affected": _TOKENS[aff],
                "detail": detail}

    if m := R_PK.match(t):
        what, aff, dirw = m.groups()
        return {**base, "rule": "pk_passive", "mechanism": _MECH.get(what, "other_pk"),
                "direction": _dir(dirw), "actor": _TOKENS[_other(aff)], "affected": _TOKENS[aff]}

    if m := R_RISK.match(t):
        detail, dirw = m.groups()
        return {**base, "rule": "risk", "mechanism": "adverse_risk", "direction": _dir(dirw),
                "actor": "unspecified", "affected": "unspecified", "detail": detail}

    if m := R_RISK2.match(t):
        detail, aff, dirw = m.groups()
        return {**base, "rule": "risk2", "mechanism": "adverse_risk", "direction": _dir(dirw),
                "actor": _TOKENS[_other(aff)], "affected": _TOKENS[aff], "detail": detail}

    return None


# --------------------------------------------------------------------------- #
# 3. Pipeline
# --------------------------------------------------------------------------- #
def build(df: pd.DataFrame) -> tuple[pd.DataFrame, Counter]:
    df = df.rename(columns={"Drug 1": "drug1_raw", "Drug 2": "drug2_raw",
                            "Interaction Description": "description"}).copy()
    df = df.dropna(subset=["drug1_raw", "drug2_raw", "description"]).reset_index(drop=True)

    records, unmatched = [], Counter()
    for d1, d2, desc in zip(df["drug1_raw"], df["drug2_raw"], df["description"]):
        t = mask_sentence(desc, d1, d2)
        parsed = parse_masked(t)
        if parsed is None:
            unmatched[t] += 1
            parsed = {"rule": "unparsed", "mechanism": "unparsed", "direction": None,
                      "actor": None, "affected": None, "detail": None,
                      "active_metabolites": False, "efficacy_loss": False}
        parsed["template"] = t
        records.append(parsed)

    out = pd.concat([df, pd.DataFrame.from_records(records)], axis=1)
    out["exposure_change"] = [
        _exposure(m, d) if pd.notna(d) else None for m, d in zip(out["mechanism"], out["direction"])
    ]
    out["interaction_class"] = [
        classify(m, d, e) if pd.notna(d) else "unparsed"
        for m, d, e in zip(out["mechanism"], out["direction"], out["exposure_change"])
    ]

    # resolve actor/affected tokens ("drug1"/"drug2") -> keep as role labels
    out["drug1"] = out["drug1_raw"].map(norm_ing)
    out["drug2"] = out["drug2_raw"].map(norm_ing)
    out["pair_key"] = [
        "||".join(sorted((a, b))) for a, b in zip(out["drug1"], out["drug2"])
    ]

    cat_cols = ["mechanism", "direction", "exposure_change", "actor", "affected",
                "interaction_class", "rule"]
    for c in cat_cols:
        out[c] = out[c].astype("category")
    out.insert(0, "row_id", range(len(out)))
    return out, unmatched


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--ddi", default="data/raw/db_drug_interactions.csv")
    ap.add_argument("--out", default="data/processed/interactions.parquet")
    ap.add_argument("--report-dir", default="reports")
    ap.add_argument("--min-parse-rate", type=float, default=0.995,
                    help="exit with code 1 if the parsed share is lower (CI guard)")
    args = ap.parse_args(argv)

    raw = pd.read_csv(args.ddi)
    out, unmatched = build(raw)

    n, n_bad = len(out), int((out["rule"] == "unparsed").sum())
    rate = 1 - n_bad / n

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)  # needs pyarrow

    rep = Path(args.report_dir)
    rep.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(unmatched.most_common(), columns=["template", "count"]).to_csv(
        rep / "unparsed_templates.csv", index=False)
    summary = {
        "rows": n,
        "parse_rate": round(rate, 5),
        "unparsed_rows": n_bad,
        "by_mechanism": out["mechanism"].value_counts().to_dict(),
        "by_class": out["interaction_class"].value_counts().to_dict(),
        "n_pairs_unordered": int(out["pair_key"].nunique()),
        "n_ingredients": int(pd.concat([out["drug1"], out["drug2"]]).nunique()),
    }
    (rep / "parse_summary.json").write_text(json.dumps(summary, indent=2, default=int))

    print(json.dumps(summary, indent=2, default=int))
    print("\n--- verification ---")
    print(f"working dir      : {Path.cwd()}")
    print(f"parse rate       : {rate:.4%}  ({n - n_bad:,} / {n:,} rows parsed, {n_bad:,} unparsed)")
    print(f"unique templates : {out['template'].nunique():,}  (unparsed templates: {len(unmatched):,})")
    for f in (Path(args.out), rep / "unparsed_templates.csv", rep / "parse_summary.json"):
        f = f.resolve()
        print(f"{'OK ' if f.exists() else 'MISSING'} {f}  ({f.stat().st_size / 1e6:.2f} MB)" if f.exists()
              else f"MISSING {f}")
    if n_bad:
        print("\nTop unparsed templates (add rules for these):")
        for t, c in unmatched.most_common(10):
            print(f"{c:>6}  {t}")
    print(f"\nSaved -> {args.out}")
    return 0 if rate >= args.min_parse_rate else 1


if __name__ == "__main__":
    sys.exit(main())