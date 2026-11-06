#import libraries
import re
import pandas as pd
from rapidfuzz import process, fuzz
from functools import lru_cache

#dicionary of synonyms
SYN = {
   "paracetamol": "acetaminophen", "albuterol": "salbutamol", "adrenaline": "epinephrine",
    "noradrenaline": "norepinephrine", "frusemide": "furosemide", "glibenclamide": "glyburide",
    "rifampin": "rifampicin", "aspirin": "acetylsalicylic acid", "acetylsalicylic": "acetylsalicylic acid",
    "vitamin d3": "cholecalciferol", "vit d3": "cholecalciferol",
    "vitamin b12": "cyanocobalamin", "vit b12": "cyanocobalamin",
    "vitamin c": "ascorbic acid", "vit c": "ascorbic acid", "ciclosporin": "cyclosporine",
    "thyroxine": "levothyroxine", "pethidine": "meperidine", "lignocaine": "lidocaine",
    "hyoscine butylbromide": "butylscopolamine",
}

#detect salts
SALTS = (r"hydrochloride|hcl|dihydrochloride|sodium|potassium|calcium|magnesium|maleate|sulfate|sulphate|"
         r"besylate|besilate|tartrate|citrate|succinate|fumarate|mesylate|phosphate|acetate|bromide|"
         r"nitrate|hydrobromide|monohydrate|dihydrate|anhydrous|trihydrate")

#find strength
_STRENGTH = r"\b\d+(\.\d+)?\s*(mg|mcg|µg|g|gm|ml|iu|i\.u\.|%|w/w|w/v|cfu)\b"

#clean strings from strength and any number or *
def _clean(x):
    x = re.sub(_STRENGTH, " ", x)
    x = re.sub(r"[*\d]", " ", x)
    return re.sub(r"\s+", " ", x).strip(" ,.-/")

#find extract name of drug without () or salts and return the synanome of the name
def candidates(raw):
    inner = re.findall(r"\((.*?)\)", raw)
    outer = re.sub(r"\(.*?\)", " ", raw)
    out = []
    for r in inner + [outer]:
        c = _clean(r)
        if c:
            out.append(c)
            out.append(re.sub(rf"\b({SALTS})\b", "", c).strip())
    mapped = [SYN[c] for c in out if c in SYN]
    return list(dict.fromkeys(out + mapped))

class Normalizer:
    def __init__(self, ddi_vocab: set[str], fuzzy_cutoff: int = 92):
        self.vocab = ddi_vocab
        self.names = sorted(ddi_vocab)
        self.cutoff = fuzzy_cutoff

    def one(self, raw: str) -> tuple[str | None, str]:
        cands = candidates(raw)
        for c in cands:
            if c in self.vocab:
                return c, "exact"
        for c in cands:
            hit = process.extractOne(c, self.names, scorer=fuzz.ratio, score_cutoff=self.cutoff)
            if hit:
                return hit[0], "fuzzy"
        return None, "none"

    def suggest(self, raw: str, k: int = 5) -> list[str]:
        c = (candidates(raw) or [raw])[0]
        return [m for m, _, _ in process.extract(c, self.names, scorer=fuzz.WRatio, limit=k)]

    def split(self, active: str) -> list[str]:
        return [p.strip() for p in str(active).split("+") if p.strip()]

    def product(self, active: str) -> dict:
        res = [(p, *self.one(p)) for p in self.split(active)]
        matched = frozenset(c for _, c, _ in res if c)
        full_key = "|".join(sorted(matched))
        return {
            "ingredients": matched,
            "unmatched": [p for p, c, _ in res if c is None],
            "n_total": len(res),
            "full_key": full_key,
        }

def build_pair_index(ddi: pd.DataFrame) -> dict:
    return {frozenset((a, b)): d for a, b, d in
            zip(ddi["Drug 1"], ddi["Drug 2"], ddi["Interaction Description"])}

def check_products(ing_a: frozenset, ing_b: frozenset, idx: dict) -> list[tuple[str, str, str]]:
    return [(a, b, idx[frozenset((a, b))]) for a in ing_a for b in ing_b
            if a != b and frozenset((a, b)) in idx]