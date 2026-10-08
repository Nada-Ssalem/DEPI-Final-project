#Coverage of eg_drugs active ingredients against the DDI vocabulary.

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz, process

from ingredients import build_vocab, split_active

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eg", default="Data/raw/eg_drugs.csv")
    ap.add_argument("--ddi", default="Data/raw/db_drug_interactions.csv")
    ap.add_argument("--top", type=int, default=50)
    ap.add_argument("--out", default="reports/missing_ingredients.csv")
    args = ap.parse_args()

    ddi = pd.read_csv(args.ddi, usecols=["Drug 1", "Drug 2"])
    eg = pd.read_csv(args.eg, usecols=["id", "name", "active"])
    eg = eg[~eg["name"].astype(str).str.strip().eq("دليل دواء مصر")]  # phantom rows

    vocab = build_vocab(ddi["Drug 1"], ddi["Drug 2"])
    eg["ingredients"] = eg["active"].map(lambda a: split_active(a, vocab))

    has = eg[eg["ingredients"].map(len) > 0].copy()
    flags = has["ingredients"].map(lambda t: [i in vocab for i in t])
    full, partial = flags.map(all), flags.map(any)

    # ingredient-level
    prod_count: Counter[str] = Counter()
    example: dict[str, str] = {}
    for name, ings in zip(has["name"], has["ingredients"]):
        for i in ings:
            prod_count[i] += 1
            example.setdefault(i, name)
    uniq = set(prod_count)
    uniq_cov = {i for i in uniq if i in vocab}

    print("=" * 60)
    print(f"DDI vocabulary size                : {len(vocab):,}")
    print(f"Products total                     : {len(eg):,}")
    print(f"Products with an active ingredient : {len(has):,} ({len(has)/len(eg):.1%})")
    print(f"  fully covered (all ingredients)  : {full.mean():.1%}")
    print(f"  partially covered (>=1)          : {partial.mean():.1%}")
    print(f"  not covered at all               : {(~partial).mean():.1%}")
    print(f"Unique ingredients                 : {len(uniq):,}")
    print(f"  covered by DDI vocab             : {len(uniq_cov):,} ({len(uniq_cov)/len(uniq):.1%})")
    print("=" * 60)

    vocab_list = sorted(vocab)
    rows = []
    for ing, n in prod_count.most_common():
        if ing in vocab:
            continue
        m = process.extractOne(ing, vocab_list, scorer=fuzz.WRatio)
        rows.append({
            "ingredient": ing,
            "n_products": n,
            "example_product": example[ing],
            "closest_in_ddi": m[0] if m else "",
            "score": round(m[1], 1) if m else 0,
        })
        if len(rows) >= args.top:
            break

    out = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"\nTop {args.top} missing ingredients (by number of products):\n")
    print(out.to_string(index=False))
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()