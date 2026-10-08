#Ingredient-name normalisation shared by eg_drugs and the DDI dataset.

#map Egyptian-market active-ingredient strings 
from __future__ import annotations

import re
import unicodedata
from typing import Iterable, Optional

import pandas as pd

# Synonyms: key = lower-case, post-_basic() form  ->  DrugBank-style canonical name.
# This is a SEED list. Extend it from reports/missing_ingredients.csv.

SYN: dict[str, str] = {
    # British / international / Egyptian-market names -> DrugBank names
    "paracetamol": "acetaminophen",
    "adrenaline": "epinephrine",
    "noradrenaline": "norepinephrine",
    "salbutamol": "albuterol",
    "frusemide": "furosemide",
    "glibenclamide": "glyburide",
    "pethidine": "meperidine",
    "lignocaine": "lidocaine",
    "hyoscine": "scopolamine",
    "glyceryl trinitrate": "nitroglycerin",
    "gtn": "nitroglycerin",
    "dipyrone": "metamizole",
    "rifampin": "rifampicin",
    "aciclovir": "acyclovir",
    "amoxycillin": "amoxicillin",
    "cephalexin": "cefalexin",
    "ciclosporin": "cyclosporine",
    "cyclosporin": "cyclosporine",
    "thyroxine": "levothyroxine",
    "l-thyroxine": "levothyroxine",
    "levothyroxine sodium": "levothyroxine",
    "benzylpenicillin": "penicillin g",
    "penicillin v": "phenoxymethylpenicillin",
    "acetylsalicylic acid": "acetylsalicylic acid",
    "aspirin": "acetylsalicylic acid",
    "asa": "acetylsalicylic acid",
    "n-acetylcysteine": "acetylcysteine",
    "valproate": "valproic acid",
    "divalproex": "valproic acid",
    "clavulanate": "clavulanic acid",
    "clavulanic": "clavulanic acid",
    "ethinylestradiol": "ethinylestradiol",
    "ethinyl estradiol": "ethinylestradiol",
    "vitamin k": "phytonadione",
    "vitamin k1": "phytonadione",
    "phytomenadione": "phytonadione",
    "orphenadrine citrate": "orphenadrine",
    "chlorphenamine": "chlorpheniramine",
    "dexchlorpheniramine": "dexchlorpheniramine",
    "mefenamic": "mefenamic acid",
    "tranexamic": "tranexamic acid",
    "folic": "folic acid",
    "ascorbic": "ascorbic acid",
    # vitamins -> chemical names
    "vitamin c": "ascorbic acid",
    "vitamin d": "cholecalciferol",
    "vitamin d3": "cholecalciferol",
    "vitamin b12": "cyanocobalamin",
    "vitamin b6": "pyridoxine",
    "vitamin b1": "thiamine",
    "vitamin b2": "riboflavin",
    "vitamin b3": "niacin",
    "nicotinamide": "niacinamide",
    "vitamin b9": "folic acid",
    "vitamin e": "tocopherol",
    "vitamin a": "retinol",
}

# Fixed-dose combo brand-generic names that must be expanded to components.
COMBO: dict[str, tuple[str, ...]] = {
    "co-trimoxazole": ("sulfamethoxazole", "trimethoprim"),
    "cotrimoxazole": ("sulfamethoxazole", "trimethoprim"),
    "co-amoxiclav": ("amoxicillin", "clavulanic acid"),
    "co-codamol": ("acetaminophen", "codeine"),
}

SALT_TOKENS: frozenset[str] = frozenset({
    "hydrochloride", "dihydrochloride", "hcl", "hydrobromide", "hbr", "bromide",
    "sodium", "potassium", "calcium", "magnesium",
    "sulfate", "sulphate", "maleate", "mesylate", "mesilate", "besylate", "besilate",
    "tartrate", "succinate", "phosphate", "acetate", "citrate", "fumarate",
    "tosylate", "napsylate", "lactate", "gluconate", "nitrate", "oxalate",
})

# If stripping salts leaves only one of these, the "salt" was really the compound.
_ANION_ONLY: frozenset[str] = frozenset({
    "chloride", "carbonate", "bicarbonate", "oxide", "hydroxide", "citrate", "lactate",
    "gluconate", "sulfate", "phosphate", "acetate",
})

_UNIT_RE = re.compile(r"\b\d+(?:[.,]\d+)?\s*(?:mg|mcg|ug|g|ml|iu|%)\b")
_NOISE_RE = re.compile(
    r"\b(?:ip|bp|usp|anhydrous|monohydrate|dihydrate|trihydrate|hemihydrate|hydrate|"
    r"micronized|micronised)\b"
)
_SPELL: list[tuple[str, str]] = [
    (r"sulph", "sulf"),
    (r"oestr", "estr"),
    (r"aluminium", "aluminum"),
]
_SPLIT_RE = re.compile(r"\s*(?:\+|&|;|\s/\s)\s*")


# Internals
def _ascii_lower(s: str) -> str:
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower().strip()


def _basic(s: str) -> str:
    #Lower-case, drop doses/pharmacopoeia tags/punctuation, unify spelling.
    s = _UNIT_RE.sub(" ", s)
    s = _NOISE_RE.sub(" ", s)
    s = re.sub(r"[^a-z0-9\- ]", " ", s)
    for pat, rep in _SPELL:
        s = re.sub(pat, rep, s)
    return re.sub(r"\s+", " ", s).strip(" -")


def _strip_salts(base: str) -> str:
    kept = [t for t in base.split() if t not in SALT_TOKENS]
    stripped = " ".join(kept)
    if not stripped or stripped in _ANION_ONLY:
        return base
    return stripped


def _candidates(raw: str) -> tuple[list[str], str]:
    #Return (ordered candidate names, fallback name when no vocabulary is given).
    s = _ascii_lower(raw)
    inner = re.findall(r"\(([^)]*)\)", s)
    outer = re.sub(r"\([^)]*\)", " ", s)

    cands: list[str] = []
    fallback = ""
    for part in [outer, *inner]:
        base = _basic(part)
        if not base:
            continue
        stripped = _strip_salts(base)
        if not fallback:
            fallback = SYN.get(stripped, stripped)
        for c in (base, stripped):
            c = SYN.get(c, c)
            if c not in cands:
                cands.append(c)
    return cands, fallback


# Public API
def canon_key(name: str) -> str:
    #Key used to build the DDI vocabulary (same cleaning, no salt stripping/synonyms).
    return _basic(_ascii_lower(name))


def build_vocab(*series: Iterable[str]) -> frozenset[str]:
    #Vocabulary of canonical names, e.g. build_vocab(ddi['Drug 1'], ddi['Drug 2']).
    out: set[str] = set()
    for s in series:
        out.update(canon_key(x) for x in pd.Series(list(s)).dropna().unique())
    out.discard("")
    return frozenset(out)


def norm_ing(s: str, vocab: Optional[frozenset[str]] = None) -> str:
    #Normalise ONE ingredient string.
    if s is None or (not isinstance(s, str) and pd.isna(s)):
        return ""
    cands, fallback = _candidates(s)
    if vocab:
        for c in cands:
            if c in vocab:
                return c
    return fallback


def split_active(a, vocab: Optional[frozenset[str]] = None) -> tuple[str, ...]:
    #Split an `active` cell ("a + b + c") into a sorted tuple of unique canonical names.
    if a is None or (not isinstance(a, str) and pd.isna(a)):
        return ()
    out: set[str] = set()
    for part in _SPLIT_RE.split(str(a)):
        if not part.strip():
            continue
        n = norm_ing(part, vocab)
        if n:
            out.update(COMBO.get(n, (n,)))
    return tuple(sorted(out))