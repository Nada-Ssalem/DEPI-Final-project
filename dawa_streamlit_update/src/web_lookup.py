"""Find the active ingredients of a medicine that is not in our data, using a web search.

Design rules (this feeds a medical lookup, so it is deliberately conservative):
  * An ingredient is accepted only if it belongs to the interaction vocabulary AND it
    appears in at least `min_sources` independent search results (or the user typed it).
  * Ingredients are looked for near anchor phrases ("active ingredient", "contains",
    "المادة الفعالة", ...) and in result titles, not anywhere in the page text.
  * Everything is returned as a *proposal*; the UI shows the sources and lets the user
    correct the ingredients, strength and form before anything is checked.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Iterable

from src.normalize import SYN
from src.parse import parse_form, parse_name

# Arabic spellings of common ingredients. Only entries whose target exists in the
# interaction vocabulary are used. Extend (and have a pharmacist review) as needed.
AR_SYN = {
    "باراسيتامول": "acetaminophen", "بارسيتامول": "acetaminophen",
    "ايبوبروفين": "ibuprofen", "إيبوبروفين": "ibuprofen",
    "اسبرين": "acetylsalicylic acid", "أسبرين": "acetylsalicylic acid",
    "ديكلوفيناك": "diclofenac",
    "اموكسيسيلين": "amoxicillin", "أموكسيسيلين": "amoxicillin",
    "ازيثرومايسين": "azithromycin", "أزيثرومايسين": "azithromycin",
    "ميتفورمين": "metformin",
    "اوميبرازول": "omeprazole", "أوميبرازول": "omeprazole",
    "سيتريزين": "cetirizine", "سيتيريزين": "cetirizine",
    "لوراتادين": "loratadine",
    "كلوبيدوجريل": "clopidogrel", "كلوبيدوغريل": "clopidogrel",
    "وارفارين": "warfarin",
    "اتورفاستاتين": "atorvastatin", "أتورفاستاتين": "atorvastatin",
    "املوديبين": "amlodipine", "أملوديبين": "amlodipine",
}

_ANCHORS = re.compile(
    r"(active\s+ingredients?|active\s+substances?|generic\s+name|composition|contains?|"
    r"المادة\s+الفعالة|المواد\s+الفعالة|المادة\s+العلمية|يحتوي\s+على|تحتوي\s+على|التركيب)"
)


class WebLookupError(RuntimeError):
    """Raised when the web search itself fails (offline, blocked, library missing)."""


@dataclass
class LookupResult:
    query: str
    ingredients: list[str] = field(default_factory=list)   # accepted, pre-selected in the UI
    weak: list[str] = field(default_factory=list)           # seen in a single result only
    evidence: dict[str, int] = field(default_factory=dict)  # ingredient -> number of sources
    strength: tuple = ()
    form: str | None = None
    sources: list[dict] = field(default_factory=list)
    n_results: int = 0


def ddg_search(query: str, max_results: int = 8) -> list[dict]:
    """Default search backend (DuckDuckGo through the `ddgs` package).

    Any function with the same signature can replace it (SerpAPI, Tavily, Google CSE...).
    """
    try:
        try:
            from ddgs import DDGS
        except ImportError:  # older package name
            from duckduckgo_search import DDGS
    except ImportError as e:
        raise WebLookupError("مكتبة البحث غير مثبتة. ثبّتها بالأمر: pip install ddgs") from e

    queries = [f"{query} دواء المادة الفعالة", f"{query} tablets active ingredient Egypt"]
    seen: set[str] = set()
    out: list[dict] = []
    try:
        engine = DDGS()
        for q in queries:
            for r in engine.text(q, max_results=max_results) or []:
                url = r.get("href") or r.get("url") or ""
                if url in seen:
                    continue
                seen.add(url)
                out.append({"title": r.get("title", ""), "snippet": r.get("body", ""), "url": url})
    except Exception as e:  # network errors, rate limits, blocked requests
        raise WebLookupError(f"تعذّر البحث في الإنترنت: {e}") from e
    return out


def _windows(text: str) -> Iterable[str]:
    t = text.lower()
    for m in _ANCHORS.finditer(t):
        seg = t[m.end(): m.end() + 120]
        yield re.split(r"[.;\n]", seg)[0]


def _mode(values, need: int, default):
    c = Counter(v for v in values if v)
    if not c:
        return default
    v, n = c.most_common(1)[0]
    return v if n >= need else default


class WebLookup:
    def __init__(
        self,
        vocab: Iterable[str],
        search_fn: Callable[[str], list[dict]] | None = None,
        min_sources: int = 2,
        max_ingredients: int = 6,
    ):
        self.vocab = frozenset(vocab)
        self.search_fn = search_fn or ddg_search
        self.min_sources = min_sources
        self.max_ingredients = max_ingredients

        self.alias = {k: v for k, v in SYN.items() if v in self.vocab and k not in self.vocab}
        self.ar = {k: v for k, v in AR_SYN.items() if v in self.vocab}
        terms = sorted({t for t in (*self.vocab, *self.alias) if len(t) >= 4}, key=len, reverse=True)
        self._rx = (
            re.compile(r"(?<![a-z0-9])(?:" + "|".join(re.escape(t) for t in terms) + r")(?![a-z0-9])")
            if terms else None
        )

    def _find(self, text: str) -> set[str]:
        t = text.lower()
        found: set[str] = set()
        if self._rx:
            for m in self._rx.finditer(t):
                s = m.group(0)
                found.add(self.alias.get(s, s))
        for k, v in self.ar.items():
            if k in t:
                found.add(v)
        return found

    def lookup(self, query: str) -> LookupResult:
        q = query.strip()
        results = self.search_fn(q) or []

        evidence: Counter = Counter()
        per_result: list[set[str]] = []
        for r in results:
            title, body = r.get("title", ""), r.get("snippet", "")
            found = self._find(title)
            for w in _windows(f"{title}. {body}"):
                found |= self._find(w)
            per_result.append(found)
            evidence.update(found)
        for ing in self._find(q):      # the user typed an ingredient name themselves
            evidence[ing] += self.min_sources

        ranked = evidence.most_common()
        accepted = [i for i, c in ranked if c >= self.min_sources][: self.max_ingredients]
        weak = [i for i, c in ranked if c < self.min_sources][:5]

        parsed = parse_name(q.lower())
        strength = tuple(parsed["strength"]) or _mode(
            [tuple(parse_name(r.get("title", "").lower())["strength"]) for r in results], 2, ()
        )
        form = parsed["form"] or _mode(
            [parse_form(r.get("title", "").lower()) for r in results], 2, None
        )

        acc = set(accepted)
        picked = [r for r, f in zip(results, per_result) if f & acc] or results[:5]
        sources, seen = [], set()
        for r in picked:
            url = r.get("url", "")
            if url and url not in seen:
                seen.add(url)
                sources.append({"title": r.get("title") or url, "url": url})
        return LookupResult(
            query=q, ingredients=accepted, weak=weak, evidence=dict(evidence),
            strength=strength, form=form, sources=sources[:6], n_results=len(results),
        )
