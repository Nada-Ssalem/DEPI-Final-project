from __future__ import annotations
import math
import streamlit as st

from src.interactions import Checker, INTERACTION, NONE_FOUND, PROCESSED
from src.alternatives import AlternativeFinder
from src.web_lookup import WebLookup, WebLookupError
from src.parse import parse_name, FORMS
from src.i18n import LEVEL_AR, FORM_AR, PACK_UNIT_AR, REASON_AR, strength_to_text, is_arabic
from src.ui import inject_css
from src import llm

st.set_page_config(page_title="دَوا | فاحص تفاعلات الأدوية", page_icon="💊", layout="centered")
inject_css()

if not (PROCESSED / "products.parquet").exists() or not (PROCESSED / "ddi_clean.parquet").exists():
    st.error("ملفات البيانات المعالجة غير موجودة. شغّل أولًا: `python -m src.preparation`")
    st.stop()


@st.cache_resource(show_spinner="جاري تحميل البيانات…")
def load():
    checker = Checker.load()
    vocab = {x for pair in checker.index for x in pair}
    return checker, AlternativeFinder(checker.products), WebLookup(vocab), sorted(vocab)


checker, finder, web, VOCAB = load()

with st.sidebar:
    st.header("الإعدادات")
    web_on = st.toggle("البحث في الإنترنت إذا لم يوجد الدواء", value=True)
    use_llm = st.toggle("شرح مبسّط بالذكاء الاصطناعي (Ollama)", value=False)
    model = st.text_input("اسم النموذج", llm.DEFAULT_MODEL, disabled=not use_llm)
    if use_llm and not llm.ollama_available():
        st.warning("Ollama غير متاح، سيُستخدم الشرح الثابت بدلًا منه.")

DISCLAIMER = "أداة تعليمية مساندة للقرار، وليست بديلًا عن الصيدلي أو الطبيب."


# ---------------------------------------------------------------- navigation
def go(service: str | None) -> None:
    st.session_state["service"] = service


def back_button() -> None:
    st.button("العودة إلى القائمة الرئيسية", key="back", on_click=go, args=(None,))


# ------------------------------------------------------------- drug picking
def _force_web(key: str, q: str) -> None:
    st.session_state[f"force_{key}"] = q


def _clear_web(key: str) -> None:
    st.session_state.pop(f"force_{key}", None)


def _retry(q: str) -> None:
    st.session_state.get("web_cache", {}).pop(q, None)


def _lookup(q: str) -> dict:
    cache = st.session_state.setdefault("web_cache", {})
    if q not in cache:
        with st.spinner("جاري البحث في الإنترنت…"):
            try:
                cache[q] = {"res": web.lookup(q)}
            except WebLookupError as e:
                cache[q] = {"error": str(e)}
            except Exception as e:  # never let a search problem break the page
                cache[q] = {"error": f"حدث خطأ غير متوقع أثناء البحث: {e}"}
    return cache[q]


def web_card(q: str, key: str) -> dict | None:
    """Show what the web search found, let the user correct it, return a product record."""
    out = _lookup(q)
    if "error" in out:
        st.error(out["error"])
        st.button("إعادة المحاولة", key=f"retry_{key}", on_click=_retry, args=(q,))
        return None
    res = out["res"]

    with st.container(border=True):
        st.markdown("**نتيجة البحث في الإنترنت**")
        st.caption("هذه بيانات غير موثّقة من نتائج بحث. راجعها وعدّلها قبل الاعتماد عليها.")
        if res.ingredients:
            st.write("المواد الفعالة المقترحة: " + "، ".join(
                f"{i} (في {res.evidence[i]} مصادر)" for i in res.ingredients))
        else:
            st.warning("لم نصل إلى مادة فعالة مؤكدة. اخترها يدويًا من القائمة إن كنت تعرفها.")
        if res.weak:
            st.caption("ظهرت في مصدر واحد فقط ولم تُختر تلقائيًا: " + "، ".join(res.weak))

        ings = st.multiselect("المواد الفعالة", VOCAB, default=res.ingredients, key=f"mi_{key}_{q}")
        s_txt = st.text_input("التركيز (مثال: 400 mg)", value=strength_to_text(res.strength),
                              key=f"ms_{key}_{q}")
        forms = [None] + [f for f, _ in FORMS]
        form = st.selectbox("الشكل الدوائي", forms,
                            index=forms.index(res.form) if res.form in forms else 0,
                            format_func=lambda f: FORM_AR.get(f, "غير معروف"), key=f"mf_{key}_{q}")
        if res.sources:
            with st.expander("المصادر"):
                for s in res.sources:
                    st.markdown(f"- [{s['title']}]({s['url']})")

    if not ings:
        st.warning("اختر مادة فعالة واحدة على الأقل لمتابعة الفحص.")
        return None
    strength = parse_name(s_txt.lower())["strength"] if s_txt.strip() else ()
    return {
        "name": f"{q} (من الإنترنت)",
        "ingredients": sorted(ings), "unmatched": [], "full_key": "|".join(sorted(ings)),
        "strength": list(strength), "form": form,
        "pack_qty": math.nan, "pack_unit": None, "unit_price": math.nan, "price": math.nan,
        "is_web": True,
    }


def drug_picker(label: str, key: str):
    """Returns a product name from our data, a web-built record, or None."""
    q = st.text_input(label, key=f"q_{key}", placeholder="مثال: brufen أو بنادول").strip()
    if not q:
        return None
    names, good = ([], False) if is_arabic(q) else checker.search_ex(q)
    forced = st.session_state.get(f"force_{key}") == q

    if good and not forced:
        choice = st.selectbox("اختر الدواء من القائمة", names, key=f"s_{key}")
        if web_on:
            st.button("دوائي غير موجود في القائمة، ابحث عنه في الإنترنت",
                      key=f"web_btn_{key}", on_click=_force_web, args=(key, q))
        return choice

    if not web_on:
        st.warning(f"«{q}» غير موجود في قاعدة البيانات، والبحث في الإنترنت متوقف من الإعدادات.")
        return None
    if forced and names:
        st.button("العودة إلى قائمة قاعدة البيانات", key=f"db_btn_{key}", on_click=_clear_web, args=(key,))
    elif not forced:
        st.info(f"«{q}» غير موجود في قاعدة البيانات، لذلك نبحث عنه في الإنترنت.")
    return web_card(q, key)


def label_of(p) -> str:
    return p if isinstance(p, str) else p["name"]


def is_web(p) -> bool:
    return not isinstance(p, str)


def signature(p):
    return p if isinstance(p, str) else ("web", p["name"], tuple(p["ingredients"]))


# ------------------------------------------------------------------- pages
def landing() -> None:
    st.title("دَوا")
    st.markdown('<p class="lead">فاحص التفاعلات الدوائية والبدائل الأرخص للأدوية المصرية. '
                'اختر الخدمة التي تريدها.</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True, key="card_interaction"):
            st.subheader("التفاعل بين دوائين")
            st.write("اعرف هل يوجد تفاعل مسجّل بين دوائين، وما نوع تأثيره، وهل يتكرر فيهما نفس المادة الفعالة.")
            st.button("افحص التفاعل", key="go_ix", type="primary", use_container_width=True,
                      on_click=go, args=("interaction",))
    with c2:
        with st.container(border=True, key="card_alternatives"):
            st.subheader("بديل أرخص لدواء")
            st.write("اكتب اسم دواء لتعرف الأدوية التي لها نفس المادة الفعالة والتركيز والشكل بسعر أقل.")
            st.button("ابحث عن بديل", key="go_alt", type="primary", use_container_width=True,
                      on_click=go, args=("alternatives",))


def show_interaction(res: dict, a, b) -> None:
    st.subheader("نتيجة الفحص")
    any_web = is_web(a) or is_web(b)
    if res["status"] == INTERACTION:
        st.error(f"يوجد تفاعل مسجّل بين الدوائين. أعلى درجة انتباه: {LEVEL_AR[res['max_level']]}")
    elif res["status"] == NONE_FOUND and not any_web:
        st.success("لا يوجد تفاعل مسجّل في قاعدة البيانات بين المواد الفعالة لهذين الدوائين.")
    elif res["status"] == NONE_FOUND:
        st.info("لم يُسجَّل تفاعل بين المواد الفعالة التي أكدتها. لكن بيانات أحد الدوائين جاءت من "
                "الإنترنت، فتأكد من اكتمال قائمة موادها الفعالة.")
    else:
        st.info("الفحص غير مكتمل: بعض المواد الفعالة غير موجودة في قاعدة بيانات التفاعلات، "
                "فلا يمكن الحكم.")

    if res["shared_ingredients"]:
        st.warning("تنبيه تكرار علاجي: الدوائان يحتويان على نفس المادة الفعالة: "
                   + "، ".join(res["shared_ingredients"]))

    if res["interactions"]:
        st.table([
            {"المادة الأولى": i["a"], "المادة الثانية": i["b"],
             "التأثير": i["mechanism"], "درجة الانتباه": i["level_ar"]}
            for i in res["interactions"]
        ])
        with st.expander("النص الأصلي من قاعدة البيانات (بالإنجليزية)"):
            for i in res["interactions"]:
                st.write(f"**{i['a']} + {i['b']}**: {i['description']}")
        with st.expander("شرح مبسّط", expanded=True):
            with st.spinner("جاري إعداد الشرح…"):
                text = (llm.explain(res, label_of(a), label_of(b), model)
                        if use_llm else llm.template_explanation(res))
            st.markdown(text)

    for name, miss in res["unknown"].items():
        if miss:
            st.caption(f"مواد غير موجودة في قاعدة التفاعلات لـ «{name}»: " + "، ".join(miss))


def page_interaction() -> None:
    back_button()
    st.header("التفاعل بين دوائين")
    c1, c2 = st.columns(2)
    with c1:
        a = drug_picker("الدواء الأول", "a")
    with c2:
        b = drug_picker("الدواء الثاني", "b")
    ready = a is not None and b is not None

    if st.button("افحص التفاعل", type="primary", disabled=not ready):
        st.session_state["ix"] = (signature(a), signature(b), checker.check(a, b))
    saved = st.session_state.get("ix")
    if ready and saved and saved[0] == signature(a) and saved[1] == signature(b):
        show_interaction(saved[2], a, b)


def pack_text(qty, unit) -> str:
    if qty is None or (isinstance(qty, float) and math.isnan(qty)):
        return "—"
    return f"{qty:g} {PACK_UNIT_AR.get(unit, unit)}"


def page_alternatives() -> None:
    back_button()
    st.header("بديل أرخص لدواء")
    p = drug_picker("اسم الدواء", "alt")
    if p is None:
        return
    alt = finder.find(p)
    st.subheader("البدائل")
    if not alt["alternatives"]:
        st.write(REASON_AR.get(alt["code"], alt["reason"]))
        return
    if is_web(p):
        st.info("سعر هذا الدواء غير معروف لأن بياناته من الإنترنت، فنعرض الأدوية المطابقة في السوق "
                "المصري مرتبة من الأرخص حسب سعر الوحدة. قارن حجم العبوة قبل الاختيار.")
    st.table([
        {"البديل": x["name"], "السعر": x["price"], "حجم العبوة": pack_text(x["pack_qty"], x["pack_unit"]),
         "نسبة التوفير %": "—" if x["saving_pct"] is None else x["saving_pct"]}
        for x in alt["alternatives"]
    ])
    st.caption("نفس المواد الفعالة والتركيز والشكل الدوائي. التطابق لا يضمن التكافؤ الحيوي، "
               "فأكّد البديل مع الصيدلي.")


# -------------------------------------------------------------------- router
service = st.session_state.get("service")
if service == "interaction":
    page_interaction()
elif service == "alternatives":
    page_alternatives()
else:
    landing()

st.markdown(f'<p class="note">{DISCLAIMER}</p>', unsafe_allow_html=True)
