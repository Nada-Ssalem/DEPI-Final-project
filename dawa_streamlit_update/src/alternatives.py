from __future__ import annotations
import pandas as pd


def _missing(v) -> bool:
    return v is None or (isinstance(v, float) and pd.isna(v))


class AlternativeFinder:
    def __init__(self, products: pd.DataFrame):
        self.p = products.reset_index(drop=True)
        self._by_name = {n: i for i, n in enumerate(self.p["name"])}
        self.groups = self.p.groupby("full_key").indices

    @staticmethod
    def _fail(code: str, reason: str) -> dict:
        return {"alternatives": [], "reason": reason, "code": code}

    def find(self, product, top_n: int = 5) -> dict:
        """`product` is a name from our data, or a record (dict / Series) such as one built from
        a web lookup. A record flagged is_web=True has no price, so we cannot say which
        alternative is cheaper than it: we list the equivalent products by unit price instead."""
        if isinstance(product, str):
            idx = self._by_name.get(product.lower().strip())
            if idx is None:
                return self._fail("not_found", "Product not found")
            r = self.p.iloc[idx]
        else:
            r = product

        is_web = bool(r["is_web"]) if "is_web" in r else False

        if not list(r["strength"]):
            return self._fail("no_strength", "Concentration not mentioned in name, cannot verify match")
        priced = not pd.isna(r["unit_price"])
        if _missing(r["form"]) or (not is_web and not priced):
            return self._fail("no_pack_or_form", "Unknown package size or dosage form")
        if not r["full_key"]:
            return self._fail("no_ingredients", "No recognized active ingredients, cannot find alternatives")

        cand = self.p.iloc[self.groups.get(r["full_key"], [])]
        target_strength = set(r["strength"])
        mask = (
            (cand["name"] != r["name"])
            & (cand["form"] == r["form"])
            & (cand["strength"].map(lambda s: set(s) == target_strength))
        )
        if is_web:
            mask &= cand["unit_price"].notna()
        else:
            mask &= (cand["pack_unit"] == r["pack_unit"]) & (cand["unit_price"] < r["unit_price"])
        cand = cand[mask].sort_values("unit_price").head(top_n)

        out = [
            {
                "name": c["name"],
                "price": float(c["price"]),
                "unit_price": float(c["unit_price"]),
                "pack_qty": None if pd.isna(c["pack_qty"]) else float(c["pack_qty"]),
                "pack_unit": c["pack_unit"],
                "saving_pct": None if is_web else float(round(100 * (1 - c["unit_price"] / r["unit_price"]), 1)),
            }
            for _, c in cand.iterrows()
        ]
        if out:
            return {"alternatives": out, "reason": None, "code": None}
        if is_web:
            return self._fail("no_match_web", "No product in our data has the same ingredients, strength and form")
        return self._fail("no_cheaper", "No cheaper alternative found with the same concentration and package size")
