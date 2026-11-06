from __future__ import annotations
from pathlib import Path
import pandas as pd
from rapidfuzz import process, fuzz
from src.severity import LEVEL

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"

INTERACTION = "interaction"
NONE_FOUND  = "none_found"
INCOMPLETE  = "incomplete"


class Checker:
    def __init__(self, products: pd.DataFrame, ddi: pd.DataFrame):
        self.products = products.reset_index(drop=True)
        self.names = self.products["name"].tolist()
        self._by_name = {n: i for i, n in enumerate(self.names)}

        self.index = {
            frozenset((a, b)): (d, int(l), m)
            for a, b, d, l, m in zip(
                ddi["Drug 1"], ddi["Drug 2"], ddi["Interaction Description"],
                ddi["level"], ddi["mechanism_ar"],
            )
        }

    @classmethod
    def load(cls, folder: Path = PROCESSED) -> "Checker":
        return cls(
            pd.read_parquet(folder / "products.parquet"),
            pd.read_parquet(folder / "ddi_clean.parquet"),
        )

    def search(self, query: str, k: int = 8) -> list[str]:
        q = query.lower().strip()
        if not q:
            return []
        starts = [n for n in self.names if n.startswith(q)][:k]
        if len(starts) >= k:
            return starts
        fuzzy = [m for m, _, _ in process.extract(q, self.names, scorer=fuzz.WRatio, limit=k * 2)]
        return list(dict.fromkeys(starts + fuzzy))[:k]

    def product(self, name: str) -> pd.Series:
        return self.products.iloc[self._by_name[name.lower().strip()]]

    def check(self, name_a: str, name_b: str) -> dict:
        a, b = self.product(name_a), self.product(name_b)
        ia, ib = list(a["ingredients"]), list(b["ingredients"])
        found = []
        for x in ia:
            for y in ib:
                if x != y and frozenset((x, y)) in self.index:
                    desc, lvl, mech = self.index[frozenset((x, y))]
                    found.append({
                        "a": x,
                        "b": y,
                        "description": desc,
                        "level_num": lvl,      
                        "level_ar": LEVEL[lvl],
                        "mechanism": mech,
                    })
        found.sort(key=lambda r: -r["level_num"])

        shared = sorted(set(ia) & set(ib))
        unknown = {name_a: list(a["unmatched"]), name_b: list(b["unmatched"])}
        has_unknown = any(unknown.values()) or not ia or not ib

        if found:
            status = INTERACTION
        elif has_unknown:
            status = INCOMPLETE
        else:
            status = NONE_FOUND

        return {
            "status": status,
            "interactions": found,
            "shared_ingredients": shared,
            "unknown": unknown,
            "ingredients": {name_a: ia, name_b: ib},
            "max_level": max((r["level_num"] for r in found), default=0),
        }