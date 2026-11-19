from __future__ import annotations
from pathlib import Path
import pandas as pd
from rapidfuzz import process, fuzz
from src.i18n import LEVEL_AR, mech_ar

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

    def search_ex(self, query: str, k: int = 8, min_score: float = 88) -> tuple[list[str], bool]:
        """Like search(), but also says whether the query really matches something in our data.

        fuzzy search always returns the k closest names, even for a drug we do not have, so the
        interface needs this flag to decide when to fall back to a web search.
        """
        q = query.lower().strip()
        if not q:
            return [], False
        starts = [n for n in self.names if n.startswith(q)][:k]
        ext = process.extract(q, self.names, scorer=fuzz.WRatio, limit=k * 2)
        best = ext[0][1] if ext else 0
        names = list(dict.fromkeys(starts + [m for m, _, _ in ext]))[:k]
        return names, bool(starts) or best >= min_score

    def product(self, name: str) -> pd.Series:
        return self.products.iloc[self._by_name[name.lower().strip()]]

    def _resolve(self, x):
        """A product is either a name from our data or a ready-made record (dict / Series),
        for example one built from a web lookup."""
        return self.product(x) if isinstance(x, str) else x

    @staticmethod
    def _label(x) -> str:
        return x if isinstance(x, str) else str(x["name"])

    def check(self, name_a, name_b) -> dict:
        a, b = self._resolve(name_a), self._resolve(name_b)
        la, lb = self._label(name_a), self._label(name_b)
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
                        "level_ar": LEVEL_AR[lvl],
                        "mechanism": mech_ar(mech),
                    })
        found.sort(key=lambda r: -r["level_num"])

        shared = sorted(set(ia) & set(ib))
        unknown = {la: list(a["unmatched"]), lb: list(b["unmatched"])}
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
            "ingredients": {la: ia, lb: ib},
            "max_level": max((r["level_num"] for r in found), default=0),
        }
