import streamlit as st

from src.alternatives import AlternativeFinder
from src.gemini_service import identify_medicine
from src.interactions import Checker
from src.llm import explain

st.set_page_config(
    page_title="Dawa — Drug Alternatives & Interaction Checker",
    page_icon="💊",
    layout="wide",
)

if "screen" not in st.session_state:
    st.session_state.screen = "home"


@st.cache_resource
def load_services():
    checker = Checker.load()
    finder = AlternativeFinder(checker.products)
    return checker, finder


checker, finder = load_services()


def go_to(screen_name: str):
    st.session_state.screen = screen_name


def render_gemini_fallback(
    state_key: str,
    medicine_name: str | None = None,
    uploaded_file=None,
    title: str = "Not found locally? Identify with Gemini (Name or Image)",
):
    result_key = f"{state_key}_result"

    with st.expander(title, expanded=False):
        if uploaded_file is not None:
            st.image(
                uploaded_file,
                caption="Uploaded medicine package",
                use_container_width=True,
            )

        clean_name = (
            medicine_name.strip()
            if isinstance(medicine_name, str) and medicine_name.strip()
            else None
        )

        if not clean_name and uploaded_file is None:
            st.caption("Enter a medicine name above or upload a package image first.")

        if st.button("Identify Medicine", key=f"{state_key}_btn"):
            image_bytes = (
                uploaded_file.getvalue() if uploaded_file is not None else None
            )
            mime_type = (
                uploaded_file.type if uploaded_file is not None else None
            )

            with st.spinner("Identifying medicine via Gemini..."):
                st.session_state[result_key] = identify_medicine(
                    medicine_name=clean_name,
                    image_bytes=image_bytes,
                    mime_type=mime_type,
                )

        res = st.session_state.get(result_key)
        if not res:
            return

        status = res.get("status")

        if status == "identified":
            st.success(
                f"Identified: **{res.get('brand_name') or clean_name or 'Medicine'}** "
                f"(Confidence: {res.get('confidence', 0):.0%})"
            )

            ingredients = res.get("active_ingredients") or []
            strengths = res.get("strength") or []
            dosage_form = res.get("dosage_form") or "Not specified"
            approx_price = res.get("approximate_price") or "Not available / varies"

            st.write("**Active Ingredients:**", ", ".join(ingredients) if ingredients else "—")
            st.write("**Strength:**", ", ".join(strengths) if strengths else "—")
            st.write("**Dosage Form:**", dosage_form)
            st.write("**Approximate Price:**", approx_price)

            if res.get("reason"):
                st.caption(res["reason"])

            st.warning(
                "Informational/guidance only — approximate prices may vary, "
                "and active ingredients/dosage must be verified with a licensed pharmacist or physician."
            )

        elif status == "unknown":
            st.info(
                res.get("reason")
                or "Could not identify the medicine with sufficient confidence."
            )

        else:
            st.error(
                res.get("reason")
                or "An error occurred while identifying the medicine."
            )


def render_home_screen():
    st.title("💊 Dawa")
    st.caption(
        "Check potential drug–drug interactions and explore local medicine alternatives."
    )
    st.warning(
        "For educational and informational purposes only. Always consult a doctor or pharmacist before changing or combining medications."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Find Alternatives")
        st.write(
            "Search for a medication and explore alternatives sharing active ingredients, with online AI lookup fallback."
        )
        if st.button("Open Alternative Finder", use_container_width=True, key="go_alt"):
            go_to("alternatives")

    with col2:
        st.subheader("Check Interactions")
        st.write(
            "Select two medications to inspect potential interactions, with online AI identification for unknown items."
        )
        if st.button("Open Interaction Checker", use_container_width=True, key="go_ddi"):
            go_to("interactions")


def render_alternatives_screen():
    if st.button("← Back to Home", key="back_from_alt"):
        go_to("home")
        st.rerun()

    st.header("Find Medicine Alternatives")

    query = st.text_input(
        "Enter medicine name",
        placeholder="e.g., Brufen, Panadol, Augmentin, Congestal...",
        key="alt_query",
    )
    uploaded_img = st.file_uploader(
        "Upload package image (optional)",
        type=["jpg", "jpeg", "png", "webp"],
        key="alt_img",
    )

    clean_name = query.strip() if isinstance(query, str) and query.strip() else None
    img_bytes = uploaded_img.getvalue() if uploaded_img is not None else None
    mime_type = uploaded_img.type if uploaded_img is not None else None

    if clean_name or img_bytes:
        request_sig = (
            clean_name,
            mime_type,
            len(img_bytes) if img_bytes else 0,
        )
        if st.session_state.get("alt_auto_gemini_sig") != request_sig:
            with st.spinner("Identifying medicine via Gemini..."):
                st.session_state["alt_auto_gemini_res"] = identify_medicine(
                    medicine_name=clean_name,
                    image_bytes=img_bytes,
                    mime_type=mime_type,
                )
            st.session_state["alt_auto_gemini_sig"] = request_sig

        gemini_res = st.session_state.get("alt_auto_gemini_res")
        with st.expander("Gemini Identification Result", expanded=True):
            if uploaded_img is not None:
                st.image(
                    uploaded_img,
                    caption="Uploaded medicine package",
                    use_container_width=True,
                )

            if gemini_res:
                status = gemini_res.get("status")
                if status == "identified":
                    st.success(
                        f"Identified: **{gemini_res.get('brand_name') or clean_name or 'Medicine'}** "
                        f"(Confidence: {gemini_res.get('confidence', 0):.0%})"
                    )
                    ingredients = gemini_res.get("active_ingredients") or []
                    strengths = gemini_res.get("strength") or []
                    dosage_form = gemini_res.get("dosage_form") or "Not specified"
                    approx_price = gemini_res.get("approximate_price") or "Not available / varies"

                    st.write("**Active Ingredients:**", ", ".join(ingredients) if ingredients else "—")
                    st.write("**Strength:**", ", ".join(strengths) if strengths else "—")
                    st.write("**Dosage Form:**", dosage_form)
                    st.write("**Approximate Price:**", approx_price)

                    if gemini_res.get("reason"):
                        st.caption(gemini_res["reason"])

                    st.warning(
                        "Informational/guidance only — approximate prices may vary, "
                        "and active ingredients/dosage must be verified with a licensed pharmacist or physician."
                    )
                elif status == "unknown":
                    st.info(
                        gemini_res.get("reason")
                        or "Could not identify the medicine with sufficient confidence."
                    )
                else:
                    st.error(
                        gemini_res.get("reason")
                        or "An error occurred while identifying the medicine."
                    )

    st.divider()

    matches = checker.search(clean_name) if clean_name else []
    if matches:
        selected_drug = st.selectbox(
            "Matching medicines in local database",
            options=matches,
            key="alt_selected_drug",
        )
        if st.button("Check Alternatives", key="run_alt_btn"):
            with st.spinner("Searching for alternatives..."):
                alt_res = finder.find(selected_drug)

            items = alt_res.get("alternatives", [])

            if items:
                st.dataframe(items, use_container_width=True)
            else:
                st.info(
                    "Local package data is incomplete for this item; "
                    "fetching alternative suggestions via Gemini..."
                )
                with st.spinner("Finding alternatives via Gemini..."):
                    alt_gemini_res = identify_medicine(
                        medicine_name=f"alternatives with same active ingredients for {selected_drug}"
                    )

                if alt_gemini_res.get("status") == "identified":
                    st.success(
                        f"Reference / Suggested Info: **{alt_gemini_res.get('brand_name') or selected_drug}**"
                    )
                    ingredients = alt_gemini_res.get("active_ingredients") or []
                    strengths = alt_gemini_res.get("strength") or []
                    dosage_form = alt_gemini_res.get("dosage_form") or "Not specified"
                    approx_price = alt_gemini_res.get("approximate_price") or "Not available / varies"

                    st.write("**Active Ingredients:**", ", ".join(ingredients) if ingredients else "—")
                    st.write("**Strength:**", ", ".join(strengths) if strengths else "—")
                    st.write("**Dosage Form:**", dosage_form)
                    st.write("**Approximate Price:**", approx_price)

                    if alt_gemini_res.get("reason"):
                        st.write("**Notes & Alternatives Guidance:**", alt_gemini_res["reason"])

                    st.warning(
                        "Informational/guidance only — verify active ingredients, dosage, "
                        "and available alternatives with a licensed pharmacist or physician."
                    )
                else:
                    st.warning(
                        "Could not retrieve alternative suggestions right now. "
                        "Please consult a pharmacist with the active ingredients shown above."
                    )


def render_interactions_screen():
    if st.button("← Back to Home", key="back_from_ddi"):
        go_to("home")
        st.rerun()

    st.header("Drug–Drug Interaction Checker")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("First Medicine")
        q_a = st.text_input(
            "Medicine A name",
            placeholder="e.g., Aspirin",
            key="ddi_q_a",
        )
        img_a = st.file_uploader(
            "Medicine A image (optional)",
            type=["jpg", "jpeg", "png", "webp"],
            key="ddi_img_a",
        )
        matches_a = checker.search(q_a.strip()) if q_a and q_a.strip() else []
        drug_a = (
            st.selectbox("Select Medicine A", options=matches_a, key="ddi_sel_a")
            if matches_a
            else None
        )
        if not matches_a:
            render_gemini_fallback(
                state_key="ddi_a_gemini",
                medicine_name=q_a,
                uploaded_file=img_a,
                title="Medicine A not found locally? Identify with Gemini",
            )

    with col_b:
        st.subheader("Second Medicine")
        q_b = st.text_input(
            "Medicine B name",
            placeholder="e.g., Ibuprofen",
            key="ddi_q_b",
        )
        img_b = st.file_uploader(
            "Medicine B image (optional)",
            type=["jpg", "jpeg", "png", "webp"],
            key="ddi_img_b",
        )
        matches_b = checker.search(q_b.strip()) if q_b and q_b.strip() else []
        drug_b = (
            st.selectbox("Select Medicine B", options=matches_b, key="ddi_sel_b")
            if matches_b
            else None
        )
        if not matches_b:
            render_gemini_fallback(
                state_key="ddi_b_gemini",
                medicine_name=q_b,
                uploaded_file=img_b,
                title="Medicine B not found locally? Identify with Gemini",
            )

    st.divider()

    if drug_a and drug_b:
        if st.button("Check Interaction", key="run_ddi_btn"):
            with st.spinner("Checking interaction..."):
                result = checker.check(drug_a, drug_b)

            status = result.get("status")
            interactions = result.get("interactions") or []
            shared = result.get("shared_ingredients") or []
            unknown = result.get("unknown") or {}

            if status == "interaction":
                st.error("There is a documented drug interaction between the two medications.")
                for item in interactions:
                    st.write(
                        f"• **{item.get('a')}**with**{item.get('b')}** "
                        f"(severatery level: {item.get('level_ar') or item.get('level_num')}): "
                        f"{item.get('mechanism') or item.get('description')}"
                    )
            elif status == "none_found":
                st.success("No drug interaction between the two medications has been recorded in the database.")
            else:
                st.info("A complete assessment cannot be made because some ingredients are not in the database.")
                unknown_items = [
                    ing for ings in unknown.values() if isinstance(ings, list) for ing in ings
                ]
                if unknown_items:
                    st.caption("Mismatched components: " + "، ".join(unknown_items))

            if shared:
                st.warning(
                    "Note: Both medications contain the same active ingredient: "
                    + "، ".join(shared)
                )

            try:
                with st.spinner("Generating explanation..."):
                    explanation = explain(result, drug_a, drug_b)
                st.markdown("### Summary Explanation")
                st.write(explanation)
            except Exception as exc:
                st.caption(f"Explanation unavailable: {exc}")
    else:
        st.caption(
            "Select both Medicine A and Medicine B from local matches to run the interaction check."
        )


if st.session_state.screen == "home":
    render_home_screen()
elif st.session_state.screen == "alternatives":
    render_alternatives_screen()
elif st.session_state.screen == "interactions":
    render_interactions_screen()