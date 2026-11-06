import re

NUM = r"\d+(?:\.\d+)?"
_MASS = {"mg": 1.0, "g": 1000.0, "gm": 1000.0, "mcg": 0.001, "µg": 0.001, "ug": 0.001}


def _w(words: str) -> str:
    return rf"(?<![a-z])(?:{words})(?![a-z])"


FORMS = [
    ("injection", _w(r"vials?|amps?|ampoules?|inj|injection|syringes?|pens?|cartridges?|infusion|i\.v\.?|i\.m\.?")),
    ("eye_drops", _w(r"eye|ophth\w*")),
    ("nasal", _w(r"nasal|nose")),
    ("tablet", _w(r"tabs?|tablets?|lactabs?|chewable|caplets?")),
    ("capsule", _w(r"caps?|capsules?|softgels?")),
    ("suspension", _w(r"susp|suspension")),
    ("syrup", _w(r"syrup|syr|elixir")),
    ("drops", _w(r"drops?")),
    ("suppository", _w(r"supp|suppositor\w+|supps")),
    ("cream", _w(r"cream")),
    ("ointment", _w(r"oint|ointment")),
    ("gel", r"(?<![a-z])(?:emul)?gel(?![a-z])"),
    ("lotion", _w(r"lotion|emulsion")),
    ("shampoo", _w(r"shampoo")),
    ("spray", _w(r"spray|mouthspray\w*")),
    ("inhaler", _w(r"inhaler|puffs?|evohaler|turbuhaler|nebuli[sz]er|respules?")),
    ("sachet", _w(r"sachets?|packets?|stick")),
    ("solution", _w(r"solution|mouthwash|gargle|douche|enema")),
    ("powder", _w(r"powder|granules?")),
    ("patch", _w(r"patch(?:es)?")),
]
SEMISOLID = {"cream", "ointment", "gel", "lotion", "shampoo", "powder", "patch"}

_CONC = re.compile(rf"({NUM})\s*(mg|mcg|µg|ug|gm|g)\s*/\s*({NUM})?\s*(ml|l)(?![a-z])")
_COMB = re.compile(rf"((?:{NUM}\s*/\s*)+{NUM})\s*(mg|mcg|µg|ug|gm|g|iu|i\.u\.)(?![a-z])")
# ← شيلنا "ml" من _SIMPLE عشان ميغلطش على 100ml volume
_SIMPLE = re.compile(rf"({NUM})\s*(mg|mcg|µg|ug|gm|g|iu|i\.u\.|%)(?![a-z])")
_COUNT = re.compile(
    rf"(\d+)\s*(?:[a-z]\.\s*){{0,3}}\s*(?:(?:enteric|coated|film|gastro-resistant|gastro resistant|eff\.?|effervescent|"
    rf"dispersible|sublingual|chewable|extended|release|retard|forte|orodispersible|vaginal|rectal)\s*)*"
    rf"(?:tabs?|tablets?|lactabs?|caps?|capsules?|softgels?|sachets?|packets?|vials?|amps?|ampoules?|"
    rf"supp\w*|patch(?:es)?|syringes?|pens?|stick\w*|lozenges?|chewable\w*)(?![a-z])"
)
_VOL = re.compile(rf"({NUM})\s*(ml|gm|g)(?![a-z])")


def _fmt(v: float) -> str:
    return f"{v:g}"


def _one(val: float, unit: str) -> str:
    if unit in _MASS:
        return f"mg:{_fmt(round(val * _MASS[unit], 4))}"
    if unit in ("iu", "i.u."):
        return f"iu:{_fmt(val)}"
    return f"pct:{_fmt(val)}"


def parse_form(name: str) -> str | None:
    for form, pat in FORMS:
        if re.search(pat, name):
            return form
    return None


def parse_name(name: str) -> dict:
    s = name.lower()
    form = parse_form(s)
    strengths: list[str] = []

    for m in _CONC.finditer(s):
        val, unit, den, _ = m.groups()
        mg = float(val) * _MASS[unit]
        strengths.append(f"mg/ml:{_fmt(round(mg / float(den or 1), 4))}")
    s_rest = _CONC.sub(" ", s)

    for m in _COMB.finditer(s_rest):
        nums, unit = m.groups()
        for n in re.split(r"\s*/\s*", nums):
            strengths.append(_one(float(n), unit))
    s_rest = _COMB.sub(" ", s_rest)

    pack_from_g = None
    for m in _SIMPLE.finditer(s_rest):
        val, unit = m.groups()
        if unit in ("g", "gm") and form in SEMISOLID:
            pack_from_g = float(val)
            continue
        strengths.append(_one(float(val), unit))

    pack_qty, pack_unit = None, None
    mc = _COUNT.search(s_rest)
    if mc:
        pack_qty, pack_unit = float(mc.group(1)), "count"
    else:
        for m in _VOL.finditer(s_rest):
            val, unit = m.groups()
            if unit == "ml":
                pack_qty, pack_unit = float(val), "ml"
        if pack_qty is None and pack_from_g:
            pack_qty, pack_unit = pack_from_g, "g"
    if pack_qty is None and form == "injection" and re.search(_w(r"vials?|amps?|ampoules?|syringes?|pens?"), s):
        pack_qty, pack_unit = 1.0, "count"
    return {"form": form, "strength": tuple(sorted(strengths)),
            "pack_qty": pack_qty, "pack_unit": pack_unit}