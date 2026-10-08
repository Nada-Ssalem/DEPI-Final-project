"""
src/ddi/split.py
================
Drug-disjoint evaluation splits (S1 / S2 / S3) + degree-matched pseudo-negatives.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def build_pair_table(inter: pd.DataFrame) -> pd.DataFrame:
    """interactions.parquet -> one row per UNORDERED pair, with its class set."""
    df = inter[["drug1", "drug2", "interaction_class"]].copy()
    df["interaction_class"] = df["interaction_class"].astype(str)
    df = df[df["drug1"] != df["drug2"]]
    lo = np.where(df["drug1"] <= df["drug2"], df["drug1"], df["drug2"])
    hi = np.where(df["drug1"] <= df["drug2"], df["drug2"], df["drug1"])
    df = df.assign(a=lo, b=hi)
    return (df.groupby(["a", "b"], sort=True)
              .agg(n_rows=("interaction_class", "size"),
                   classes=("interaction_class", lambda s: sorted(set(s))))
              .reset_index())


def _degree(pairs: pd.DataFrame) -> pd.Series:
    return pd.concat([pairs["a"], pairs["b"]]).value_counts()


def _weighted_pool(names, degree: pd.Series):
    if len(names) == 0:
        return None
    w = degree.reindex(names).fillna(0).to_numpy(float) + 1.0  # +1: zero-degree stay drawable
    return np.array(names), w / w.sum()


def sample_negatives(pos: pd.DataFrame, known: set, unseen: set,
                     seen_pool, unseen_pool, rng, ratio: int = 1) -> pd.DataFrame:
    """Keep one side of each positive, resample the other from a same-type pool
    (seen->seen, unseen->unseen), proportional to degree. Skips known positives."""
    out = []
    for _ in range(ratio):
        left = rng.random(len(pos)) < 0.5
        keep = np.where(left, pos["a"].to_numpy(), pos["b"].to_numpy())
        corr = np.where(left, pos["b"].to_numpy(), pos["a"].to_numpy())
        is_un = np.fromiter((c in unseen for c in corr), bool, len(corr))
        new = corr.copy()
        todo = np.arange(len(pos))
        for _try in range(30):
            for flag, pool in ((False, seen_pool), (True, unseen_pool)):
                idx = todo[is_un[todo] == flag]
                if len(idx) and pool is not None:
                    new[idx] = rng.choice(pool[0], size=len(idx), p=pool[1])
            bad = np.fromiter(
                ((k == n) or ((min(k, n), max(k, n)) in known) for k, n in zip(keep[todo], new[todo])),
                bool, len(todo))
            todo = todo[bad]
            if len(todo) == 0:
                break
        ok = np.ones(len(pos), bool)
        ok[todo] = False  # drop the few that never found a valid negative
        a = np.where(keep <= new, keep, new)[ok]
        b = np.where(keep <= new, new, keep)[ok]
        out.append(pd.DataFrame({"a": a, "b": b, "label": 0, "classes": [[] for _ in range(len(a))]}))
    return pd.concat(out, ignore_index=True).drop_duplicates(["a", "b"])


def make_splits(pairs: pd.DataFrame, unseen_frac: float = 0.15, s1_frac: float = 0.10,
                val_frac: float = 0.10, neg_ratio: int = 1, seed: int = 42) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    all_drugs = np.array(sorted(set(pairs["a"]) | set(pairs["b"])))
    
    # 1. Randomly select candidate unseen drugs
    unseen_seed = set(rng.choice(all_drugs, int(len(all_drugs) * unseen_frac), replace=False))
    
    # Candidate seen pairs (neither drug in unseen_seed)
    is_seed_un = pairs["a"].isin(unseen_seed) | pairs["b"].isin(unseen_seed)
    cand_seen_pairs = pairs[~is_seed_un].copy()
    
    # 2. Partition candidate seen pairs into train, val, s1
    perm = rng.permutation(len(cand_seen_pairs))
    cand_seen_pairs = cand_seen_pairs.iloc[perm].reset_index(drop=True)
    
    deg_counts = _degree(cand_seen_pairs).to_dict()
    
    train_indices = []
    val_indices = []
    s1_indices = []
    
    n_total = len(cand_seen_pairs)
    n_val_target = int(n_total * val_frac)
    n_s1_target = int(n_total * s1_frac)
    
    current_deg = deg_counts.copy()
    
    for idx, row in cand_seen_pairs.iterrows():
        a_drug, b_drug = row["a"], row["b"]
        # Only assign to val/s1 if removing this pair won't leave any drug with 0 degree in train
        if len(val_indices) < n_val_target and current_deg[a_drug] > 1 and current_deg[b_drug] > 1:
            val_indices.append(idx)
            current_deg[a_drug] -= 1
            current_deg[b_drug] -= 1
        elif len(s1_indices) < n_s1_target and current_deg[a_drug] > 1 and current_deg[b_drug] > 1:
            s1_indices.append(idx)
            current_deg[a_drug] -= 1
            current_deg[b_drug] -= 1
        else:
            train_indices.append(idx)
            
    train = cand_seen_pairs.iloc[train_indices].reset_index(drop=True)
    val = cand_seen_pairs.iloc[val_indices].reset_index(drop=True)
    s1 = cand_seen_pairs.iloc[s1_indices].reset_index(drop=True)
    
    # 3. Ground truth set of train drugs & unseen drugs
    tr_drugs = set(train["a"]) | set(train["b"])
    unseen_drugs = set(all_drugs) - tr_drugs
    
    # 4. Categorize ALL remaining pairs in the dataset into S1, S2, S3
    rem_pairs = pairs[is_seed_un].copy()
    
    cnt_unseen = rem_pairs["a"].isin(unseen_drugs).astype(int) + rem_pairs["b"].isin(unseen_drugs).astype(int)
    
    extra_s1 = rem_pairs[cnt_unseen == 0]
    s2 = rem_pairs[cnt_unseen == 1]
    s3 = rem_pairs[cnt_unseen == 2]
    
    if len(extra_s1) > 0:
        s1 = pd.concat([s1, extra_s1], ignore_index=True)
        
    pos = {"train": train, "val": val, "S1": s1, "S2": s2, "S3": s3}

    known = set(zip(pairs["a"], pairs["b"]))
    deg_train, deg_all = _degree(train), _degree(pairs)
    seen_names = sorted(tr_drugs)
    unseen_names = sorted(unseen_drugs)

    splits, used_neg = {}, set()
    for name in ("S3", "S2", "S1", "val", "train"):
        p = pos[name]
        deg = deg_train if name in ("train", "val") else deg_all
        neg = sample_negatives(p, known, unseen_drugs, _weighted_pool(seen_names, deg),
                               _weighted_pool(unseen_names, deg_all), rng, neg_ratio)
        neg = neg[[(x, y) not in used_neg for x, y in zip(neg["a"], neg["b"])]]
        used_neg |= set(zip(neg["a"], neg["b"]))
        p = p.assign(label=1)[["a", "b", "label", "classes"]]
        splits[name] = pd.concat([p, neg], ignore_index=True).assign(split=name)
    return {k: splits[k] for k in ("train", "val", "S1", "S2", "S3")}


def assert_no_leak(splits: dict[str, pd.DataFrame]) -> dict:
    tr = splits["train"]
    tr_drugs = set(tr["a"]) | set(tr["b"])
    rep = {}
    for n in ("val", "S1"):
        s = splits[n]
        rep[f"{n}_drugs_all_seen"] = bool(set(s["a"]).union(s["b"]) <= tr_drugs)
    for n, k in (("S2", 1), ("S3", 2)):
        s = splits[n]
        cnt = (s["a"].isin(tr_drugs).astype(int)*-1 + 1) + (s["b"].isin(tr_drugs).astype(int)*-1 + 1)
        rep[f"{n}_unseen_drug_count_ok"] = bool((cnt == k).all())
    keys = {n: set(zip(s["a"], s["b"])) for n, s in splits.items()}
    names = list(keys)
    rep["pairs_disjoint"] = all(not (keys[x] & keys[y]) for i, x in enumerate(names) for y in names[i+1:])
    assert all(rep.values()), f"LEAK CHECK FAILED: {rep}"
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inter", default="data/processed/interactions.parquet")
    ap.add_argument("--out-dir", default="data/processed/splits")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--neg-ratio", type=int, default=1)
    a = ap.parse_args(argv)

    pairs = build_pair_table(pd.read_parquet(a.inter))
    splits = make_splits(pairs, seed=a.seed, neg_ratio=a.neg_ratio)
    print("leak check:", assert_no_leak(splits))
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    for n, s in splits.items():
        s.to_parquet(out / f"{n}.parquet", index=False)
        print(f"{n:>5}: {len(s):>8,} rows | positives {int(s.label.sum()):>8,} | drugs {len(set(s.a)|set(s.b)):>5,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
