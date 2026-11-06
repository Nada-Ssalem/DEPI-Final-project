import pytest
from src.normalize import candidates
from src.parse import parse_name


def test_candidates_parentheses_and_salts():
    assert "acetaminophen" in candidates("paracetamol(acetaminophen)")
    assert "diclofenac" in candidates("diclofenac sodium")


@pytest.mark.parametrize("name,form,strength,qty,unit", [
    ("brufen 400mg 30 tab", "tablet", ("mg:400",), 30, "count"),
    ("augmentin 1 gm 14 f.c. tabs", "tablet", ("mg:1000",), 14, "count"),
    ("tendratine 2.5mg/5ml syrup 100ml", "syrup", ("mg/ml:0.5",), 100, "ml"),
    ("voltaren 1% emulgel 100 gm", "gel", ("pct:1",), 100, "g"),
])
def test_parse_name(name, form, strength, qty, unit):
    r = parse_name(name)
    assert (r["form"], r["strength"], r["pack_qty"], r["pack_unit"]) == (form, strength, qty, unit)


@pytest.fixture(scope="module")
def checker():
    from src.interactions import Checker, PROCESSED
    if not (PROCESSED / "products.parquet").exists():
        pytest.skip("processed data not found — run `python -m src.preparation` first")
    return Checker.load()


def test_known_interactions(checker):
    r = checker.check("brufen 400mg 30 tab", "aspirin protect 100mg 30 gastro-resistant tab")
    assert r["status"] == "interaction"
    r = checker.check("marevan 5mg 102 tab", "aspirin protect 100mg 30 gastro-resistant tab")
    assert any("bleeding" in i["mechanism"].lower() for i in r["interactions"])


def test_duplicate_therapy_detected(checker):
    r = checker.check("panadol extra optizorb 24 f.c. tabs", "panadol acute head cold 20 f.c.tabs")
    assert "acetaminophen" in r["shared_ingredients"]


def test_symmetry(checker):
    a, b = "brufen 400mg 30 tab", "marevan 5mg 102 tab"
    ab = {(i["a"], i["b"]) for i in checker.check(a, b)["interactions"]}
    ba = {(i["b"], i["a"]) for i in checker.check(b, a)["interactions"]}
    assert ab == ba


def test_alternatives_are_cheaper_and_same_ingredients(checker):
    from src.alternatives import AlternativeFinder
    f = AlternativeFinder(checker.products)
    orig = checker.product("brufen 400mg 30 tab")
    alts = f.find("brufen 400mg 30 tab")["alternatives"]
    assert alts
    for a in alts:
        p = checker.product(a["name"])
        assert p["full_key"] == orig["full_key"]
        assert p["unit_price"] < orig["unit_price"]