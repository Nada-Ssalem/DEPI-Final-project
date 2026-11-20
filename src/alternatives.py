from __future__ import annotations
import pandas as pd


class AlternativeFinder:
    def __init__(self, products: pd.DataFrame):
        self.p = products.reset_index(drop=True)
        self._by_name = {n: i for i, n in enumerate(self.p["name"])}
        self.groups = self.p.groupby("full_key").indices

    def find(self, name: str, top_n: int = 5) -> dict:
        key = name.lower().strip()
        idx = self._by_name.get(key)
        if idx is None:
            return {"alternatives": [], "reason": "Product not found"}
        r = self.p.iloc[idx]

        if not list(r["strength"]):
            return {"alternatives": [], "reason": "Concentration not mentioned in name, cannot verify match"}
        if pd.isna(r["unit_price"]) or r["form"] is None:
            return {"alternatives": [], "reason": "Unknown package size or dosage form"}
        if not r["full_key"]:
            return {"alternatives": [], "reason": "No recognized active ingredients, cannot find alternatives"}

        cand = self.p.iloc[self.groups.get(r["full_key"], [])]
        target_strength = set(r["strength"])
        cand = cand[
            (cand["name"] != r["name"])
            & (cand["form"] == r["form"])
            & (cand["pack_unit"] == r["pack_unit"])
            & (cand["strength"].map(lambda s: set(s) == target_strength))
            & (cand["unit_price"] < r["unit_price"])
        ].sort_values("unit_price").head(top_n)

        out = [
            {
                "name": c["name"],
                "price": float(c["price"]),
                "unit_price": float(c["unit_price"]),
                "saving_pct": float(round(100 * (1 - c["unit_price"] / r["unit_price"]), 1)),
            }
            for _, c in cand.iterrows()
        ]
        return {
            "alternatives": out,
            "reason": None if out else "No cheaper alternative found with the same concentration and package size",
        }