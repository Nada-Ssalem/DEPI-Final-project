from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path

from src.normalize import Normalizer, SYN
from src.parse import parse_name
from src.severity import rate as severity_rate

DATA = Path("data")
OUT = DATA / "processed"


def clean_ddi(ddi: pd.DataFrame) -> pd.DataFrame:
    ddi = ddi.copy()
    ddi["Drug 1"] = ddi["Drug 1"].astype(str).str.lower().str.strip()
    ddi["Drug 2"] = ddi["Drug 2"].astype(str).str.lower().str.strip()
    ddi = ddi.dropna(subset=["Drug 1", "Drug 2", "Interaction Description"])

    a = ddi["Drug 1"].values
    b = ddi["Drug 2"].values
    key = np.where(a <= b, a + "||" + b, b + "||" + a)
    ddi = ddi.loc[~pd.Series(key).duplicated()].reset_index(drop=True)

    rated = ddi["Interaction Description"].map(severity_rate)
    ddi["level"] = rated.map(lambda t: t[0])
    ddi["mechanism_ar"] = rated.map(lambda t: t[1])
    return ddi[["Drug 1", "Drug 2", "Interaction Description", "level", "mechanism_ar"]]


def build_vocab(ddi: pd.DataFrame) -> set[str]:
    names = set(ddi["Drug 1"]).union(set(ddi["Drug 2"]))
    names = {SYN.get(n, n) for n in names}
    return names


def prepare_products(eg: pd.DataFrame, norm: Normalizer) -> pd.DataFrame:
    eg = eg.copy()
    eg["name"] = eg["name"].astype(str).str.lower().str.strip()
    eg["active"] = eg["active"].astype(str).str.lower().str.strip()
    eg = eg[eg["price"] > 0].drop_duplicates("name").reset_index(drop=True)

    prod = eg["active"].map(norm.product)
    eg["ingredients"] = prod.map(lambda d: sorted(d["ingredients"]))
    eg["unmatched"] = prod.map(lambda d: d["unmatched"])
    eg["full_key"] = prod.map(lambda d: d["full_key"])
    eg["is_drug"] = eg["ingredients"].map(len) > 0

    parsed = eg["name"].map(parse_name).apply(pd.Series)
    parsed["strength"] = parsed["strength"].map(list)
    eg = pd.concat([eg, parsed], axis=1)

    eg["pack_qty"] = eg["pack_qty"].replace(0, np.nan)
    eg["unit_price"] = eg["price"] / eg["pack_qty"]
    return eg


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    ddi_raw = pd.read_csv(DATA / "ddi.csv")
    eg_raw = pd.read_csv(DATA / "eg_drugs.csv")

    ddi = clean_ddi(ddi_raw)
    ddi.to_parquet(OUT / "ddi_clean.parquet", index=False)

    vocab = build_vocab(ddi)
    norm = Normalizer(vocab)

    products = prepare_products(eg_raw, norm)
    products.to_parquet(OUT / "products.parquet", index=False)


if __name__ == "__main__":
    main()