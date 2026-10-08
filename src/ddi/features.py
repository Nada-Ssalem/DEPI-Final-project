"""
src/ddi/features.py
===================
Feature dictionary for drug pairs.

Per-drug features (symmetric pair encoding afterwards):
  * market features from eg_drugs  (in_eg, n_products, price, 7 warning flags, therapeutic-category shares)
  * graph profile from TRAIN positives only (degree, share of each interaction_class)
    -> zeros for unseen drugs (S2/S3), exactly what happens in production for new drugs.

Pair encoding (order-independent):  [f(a)+f(b) | |f(a)-f(b)| | f(a)*f(b)]
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

try:
    from src.ingredients import norm_ing, split_active   # project modules
except ImportError:  # pragma: no cover - fallbacks so the file runs standalone
    def norm_ing(x, *_, **__):
        return re.sub(r"\s+", " ", str(x).strip().lower())

    def split_active(x):
        return [norm_ing(p) for p in re.split(r"[+,/]", str(x)) if p.strip()]

WARN_COLS = ["warning_high_blood_pressure", "warning_diabetes", "warning_pregnancy",
             "warning_lactation", "warning_kidney", "warning_liver", "warning_heart"]
TOP_CATS = 30


def _price(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.strip().str.rstrip("."), errors="coerce")


def build_market_features(eg: pd.DataFrame) -> pd.DataFrame:
    """eg_drugs.csv -> one row per normalized ingredient."""
    d = eg.dropna(subset=["active"]).copy()
    d["price"] = _price(d["price"])
    d["cat"] = (d["description"].fillna("").str.lower()
                  .str.split(r"[.,]", regex=True).str[0].str.strip().replace("", np.nan))
    d["ing"] = d["active"].map(split_active)
    d = d.explode("ing").dropna(subset=["ing"]).reset_index(drop=True)

    g = d.groupby("ing")
    feat = pd.DataFrame({
        "in_eg": 1.0,
        "log_n_products": np.log1p(g.size()),
        "log_price_median": np.log1p(g["price"].median()),
    })
    feat = feat.join(g[WARN_COLS].mean())

    top = d["cat"].value_counts().head(TOP_CATS).index
    cat = (pd.crosstab(d["ing"], d["cat"].where(d["cat"].isin(top)))
             .pipe(lambda c: c.div(c.sum(axis=1).replace(0, 1), axis=0)))
    cat.columns = [f"cat_{c}" for c in cat.columns]
    return feat.join(cat).fillna(0.0)


def build_graph_profile(train_pos: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    """Per-drug degree + class shares, from TRAIN positives only (no leakage)."""
    p = train_pos[train_pos["label"] == 1][["a", "b", "classes"]]
    long = pd.concat([p[["a", "classes"]].rename(columns={"a": "drug"}),
                      p[["b", "classes"]].rename(columns={"b": "drug"})], ignore_index=True)
    deg = long.groupby("drug").size()
    ex = long.explode("classes").dropna(subset=["classes"]).reset_index(drop=True)
    share = pd.crosstab(ex["drug"], ex["classes"]).reindex(columns=classes, fill_value=0)
    share = share.div(share.sum(axis=1).replace(0, 1), axis=0)
    share.columns = [f"gp_{c}" for c in share.columns]
    return pd.DataFrame({"log_degree": np.log1p(deg)}).join(share).fillna(0.0)


def drug_table(drugs, market: pd.DataFrame, graph: pd.DataFrame) -> pd.DataFrame:
    idx = pd.Index(sorted(set(drugs)), name="drug")
    return (pd.DataFrame(index=idx).join(market).join(graph).fillna(0.0)).astype("float32")


def pair_matrix(pairs: pd.DataFrame, table: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    A = table.reindex(pairs["a"]).to_numpy(np.float32)
    B = table.reindex(pairs["b"]).to_numpy(np.float32)
    X = np.hstack([A + B, np.abs(A - B), A * B])
    cols = table.columns.tolist()
    names = [f"sum_{c}" for c in cols] + [f"absdiff_{c}" for c in cols] + [f"prod_{c}" for c in cols]
    return X, names


def build_all(splits: dict[str, pd.DataFrame], eg: pd.DataFrame, classes: list[str]):
    """Returns {split: (X, y)}, feature_names. Graph profile uses TRAIN only."""
    market = build_market_features(eg)
    graph = build_graph_profile(splits["train"], classes)
    all_drugs = set().union(*[set(s["a"]) | set(s["b"]) for s in splits.values()])
    table = drug_table(all_drugs, market, graph)
    out, names = {}, None
    for n, s in splits.items():
        X, names = pair_matrix(s, table)
        out[n] = (X, s["label"].to_numpy())
    return out, names