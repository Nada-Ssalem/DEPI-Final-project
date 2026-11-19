import math
import pandas as pd
import pytest

pytest.importorskip("rapidfuzz")

from src.web_lookup import WebLookup, WebLookupError
from src.interactions import Checker, INTERACTION, NONE_FOUND, INCOMPLETE
from src.alternatives import AlternativeFinder

VOCAB = {"ibuprofen", "acetaminophen", "acetylsalicylic acid", "warfarin"}


def fake(results):
    return lambda q: results


def r(title, body="", url=None):
    return {"title": title, "snippet": body, "url": url or f"https://x/{abs(hash(title + body))}"}


def test_ingredient_needs_two_sources():
    results = [
        r("Brufen 400 mg tablets", "Brufen contains ibuprofen. Used for pain."),
        r("Brufen - ibuprofen 400mg", "Active ingredient: ibuprofen 400 mg"),
        r("Other page", "contains warfarin in a different product"),
    ]
    res = WebLookup(VOCAB, fake(results)).lookup("brufen")
    assert res.ingredients == ["ibuprofen"]
    assert res.weak == ["warfarin"]
    assert res.evidence["ibuprofen"] >= 2


def test_ingredient_not_found_in_random_text():
    results = [r("Welcome", "ibuprofen is mentioned but with no anchor phrase here at all")] * 2
    res = WebLookup(VOCAB, fake(results)).lookup("brufen")
    assert res.ingredients == []          # no title hit, no anchor window


def test_user_typed_ingredient_is_accepted():
    res = WebLookup(VOCAB, fake([])).lookup("paracetamol 500 mg tablets")
    assert res.ingredients == ["acetaminophen"]      # via SYN
    assert res.strength == ("mg:500",)
    assert res.form == "tablet"


def test_arabic_synonym_and_anchor():
    results = [
        r("بنادول", "المادة الفعالة: باراسيتامول"),
        r("بنادول اقراص", "يحتوي على باراسيتامول 500 مجم"),
    ]
    res = WebLookup(VOCAB, fake(results)).lookup("بنادول")
    assert res.ingredients == ["acetaminophen"]


def test_strength_and_form_from_titles_need_agreement():
    results = [r("X 400 mg tablets"), r("Y 400mg tab"), r("Z 600 mg tablets")]
    res = WebLookup(VOCAB, fake(results)).lookup("xdrug")
    assert res.strength == ("mg:400",)
    assert res.form == "tablet"


def test_search_error_propagates():
    def boom(q):
        raise WebLookupError("offline")
    with pytest.raises(WebLookupError):
        WebLookup(VOCAB, boom).lookup("x")


# ---------- web records flowing through the existing checker / finder ----------
def make_data():
    prod = pd.DataFrame([
        dict(name="aspirin protect 100mg 30 tab", ingredients=["acetylsalicylic acid"], unmatched=[],
             full_key="acetylsalicylic acid", strength=["mg:100"], form="tablet",
             pack_qty=30.0, pack_unit="count", price=30.0, unit_price=1.0),
        dict(name="brufen 400mg 30 tab", ingredients=["ibuprofen"], unmatched=[], full_key="ibuprofen",
             strength=["mg:400"], form="tablet", pack_qty=30.0, pack_unit="count", price=60.0, unit_price=2.0),
        dict(name="cheapofen 400mg 20 tab", ingredients=["ibuprofen"], unmatched=[], full_key="ibuprofen",
             strength=["mg:400"], form="tablet", pack_qty=20.0, pack_unit="count", price=20.0, unit_price=1.0),
    ])
    ddi = pd.DataFrame([dict(**{"Drug 1": "ibuprofen", "Drug 2": "acetylsalicylic acid",
                                "Interaction Description": "x may increase bleeding", "level": 3,
                                "mechanism_ar": "Bleeding risk"})])
    return prod, ddi


def web_record(ings, strength=("mg:400",), form="tablet"):
    return {"name": "foo (من الإنترنت)", "ingredients": sorted(ings), "unmatched": [],
            "full_key": "|".join(sorted(ings)), "strength": list(strength), "form": form,
            "pack_qty": math.nan, "pack_unit": None, "unit_price": math.nan, "price": math.nan,
            "is_web": True}


def test_web_record_interaction_with_db_product():
    prod, ddi = make_data()
    ch = Checker(prod, ddi)
    res = ch.check(web_record(["ibuprofen"]), "aspirin protect 100mg 30 tab")
    assert res["status"] == INTERACTION
    assert res["interactions"][0]["mechanism"] == "زيادة خطر النزف"
    assert res["interactions"][0]["level_ar"] == "عالية"


def test_web_record_without_interaction_and_unknown():
    prod, ddi = make_data()
    ch = Checker(prod, ddi)
    assert ch.check(web_record(["warfarin"]), "brufen 400mg 30 tab")["status"] == NONE_FOUND
    assert ch.check(web_record([]), "brufen 400mg 30 tab")["status"] == INCOMPLETE


def test_search_ex_flags_missing_drug():
    prod, ddi = make_data()
    ch = Checker(prod, ddi)
    names, good = ch.search_ex("brufen")
    assert good and names[0].startswith("brufen")
    assert ch.search_ex("zzzzqqqq")[1] is False


def test_alternatives_for_web_record_list_equivalents_by_unit_price():
    prod, ddi = make_data()
    alt = AlternativeFinder(prod).find(web_record(["ibuprofen"]))
    assert [a["name"] for a in alt["alternatives"]] == ["cheapofen 400mg 20 tab", "brufen 400mg 30 tab"]
    assert alt["alternatives"][0]["saving_pct"] is None


def test_alternatives_web_record_needs_strength():
    prod, ddi = make_data()
    alt = AlternativeFinder(prod).find(web_record(["ibuprofen"], strength=()))
    assert alt["code"] == "no_strength"


def test_alternatives_db_product_still_cheaper_only():
    prod, ddi = make_data()
    alt = AlternativeFinder(prod).find("brufen 400mg 30 tab")
    assert [a["name"] for a in alt["alternatives"]] == ["cheapofen 400mg 20 tab"]
    assert alt["alternatives"][0]["saving_pct"] == 50.0
