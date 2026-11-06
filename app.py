import streamlit as st
from src.interactions import Checker, INTERACTION, NONE_FOUND, INCOMPLETE, PROCESSED
from src.alternatives import AlternativeFinder
from src import llm

st.set_page_config(page_title="Drug Interaction Checker", page_icon="💊", layout="centered")
st.markdown("<style>.block-container{direction:ltr;text-align:left}</style>", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading data and initializing models...")
def load():
    checker = Checker.load()
    return checker, AlternativeFinder(checker.products)


if not (PROCESSED / "products.parquet").exists() or not (PROCESSED / "ddi_clean.parquet").exists():
    st.error("Processed data files not found. Please run: `python -m src.preparation` first.")
    st.stop()

checker, finder = load()

st.title("Drug Interaction Checker")
st.caption("A supportive clinical assistant tool for Egyptian pharmaceutical products, not a replacement for a pharmacist or physician.")

with st.sidebar:
    use_llm = st.toggle("Enable LLM Plain Explanation (Ollama)", value=False)
    model = st.text_input("Model Name", llm.DEFAULT_MODEL, disabled=not use_llm)
    if use_llm and not llm.ollama_available():
        st.warning("⚠ Ollama is not available. Falling back to template explanation.")


def pick(label: str, key: str) -> str | None:
    q = st.text_input(label, key=f"q_{key}", placeholder="e.g., brufen")
    opts = checker.search(q) if q else []
    if q and not opts:
        st.caption("No results found.")
    return st.selectbox("Select Drug", opts, key=f"s_{key}") if opts else None


c1, c2 = st.columns(2)
with c1:
    drug_a = pick("First Drug", "a")
with c2:
    drug_b = pick("Second Drug", "b")

if st.button("Check Interaction", type="primary", disabled=not (drug_a and drug_b)):
    res = checker.check(drug_a, drug_b)
    st.subheader("Interaction Results")

    if res["status"] == INTERACTION:
        st.error(f"Interaction detected between the two drugs! (Max Severity Level: {res['max_level']})")
    elif res["status"] == NONE_FOUND:
        st.success("No registered interaction found in database between these active ingredients.")
    else:
        st.info("ℹ Incomplete check: Some ingredients are missing from the interaction database.")

    if res["shared_ingredients"]:
        st.warning(f"Therapeutic Duplication Warning - Both drugs contain: {', '.join(res['shared_ingredients'])}")

    if res["interactions"]:
        st.table([
            {
                "Drug 1": i["a"],
                "Drug 2": i["b"],
                "Effect": i["mechanism"],
                "Severity": i["level_ar"],
            }
            for i in res["interactions"]
        ])
        with st.expander("View Raw Database Text"):
            for i in res["interactions"]:
                st.write(f"**{i['a']} + {i['b']}**: {i['description']}")

        with st.expander("AI Clinical Explanation", expanded=True):
            with st.spinner("Generating AI explanation..."):
                text = llm.explain(res, drug_a, drug_b, model) if use_llm else llm.template_explanation(res)
                st.markdown(text)

    for name, miss in res["unknown"].items():
        if miss:
            st.caption(f"ℹ Missing ingredients for '{name}': {', '.join(miss)}")

st.subheader("Cheaper Alternatives")
for name in (drug_a, drug_b):
    if name:
        alt = finder.find(name)
        with st.expander(f"Alternatives for {name}", expanded=bool(alt["alternatives"])):
            if alt["alternatives"]:
                st.table([
                    {
                        "Alternative Drug": a["name"],
                        "Price": a["price"],
                        "Saving %": a["saving_pct"],
                    }
                    for a in alt["alternatives"]
                ])
                st.caption("Note: Same active ingredients, pharmaceutical form, strength, and unit price comparison.")
            else:
                st.write(alt["reason"])