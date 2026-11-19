"""Page styling: right-to-left layout and the Thmanyah font."""
from __future__ import annotations
from pathlib import Path
from urllib.parse import quote
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
FONT_DIR = ROOT / "static" / "fonts"
FORMATS = {".otf": "opentype", ".ttf": "truetype", ".woff2": "woff2", ".woff": "woff"}
# order matters: "semibold"/"extrabold" must be tested before "bold"
WEIGHTS = [("thin", 100), ("extralight", 200), ("ultralight", 200), ("light", 300),
           ("semibold", 600), ("demibold", 600), ("extrabold", 800), ("heavy", 800),
           ("black", 900), ("bold", 700), ("medium", 500), ("regular", 400)]
FALLBACK = '"Segoe UI", Tahoma, "Noto Naskh Arabic", Arial, sans-serif'


def _font_files() -> list[Path]:
    files = [p for p in sorted(FONT_DIR.glob("*"))
             if p.suffix.lower() in FORMATS and "italic" not in p.name.lower()]
    sans = [p for p in files if "sans" in p.name.lower()]
    return sans or files


def font_face_css() -> str:
    rules = []
    for p in _font_files():
        key = p.stem.lower().replace("-", "").replace("_", "").replace(" ", "")
        weight = next((w for name, w in WEIGHTS if name in key), 400)
        rules.append(
            '@font-face{font-family:"Thmanyah";'
            f'src:url("app/static/fonts/{quote(p.name)}") format("{FORMATS[p.suffix.lower()]}");'
            f"font-weight:{weight};font-style:normal;font-display:swap;}}"
        )
    return "\n".join(rules)


CSS = """
:root{--ink:#14201f;--muted:#5b6b69;--line:#d3dedb;--brand:#0b5d5b;--soft:#e6f0ee;}
html, body, .stApp{direction:rtl;}
.stApp, .stApp :is(p, li, label, h1, h2, h3, h4, h5, h6, a, input, textarea, button, th, td, caption,
  [data-baseweb="select"] *, [data-baseweb="popover"] li){
  font-family:"Thmanyah", __FALLBACK__ !important;
}
.stApp :is(p, li, label, h1, h2, h3, h4, th, td, caption){text-align:right;}
.stApp :is(h1, h2, h3){font-weight:700; letter-spacing:0;}
.stApp h1{font-size:2.6rem; margin-bottom:.2rem;}
.stApp :is(input, textarea){unicode-bidi:plaintext; text-align:start;}
.stApp table :is(th, td){text-align:right;}
.block-container{max-width:760px; padding-top:2.2rem;}
footer{visibility:hidden;}
.st-key-card_interaction, .st-key-card_alternatives{
  border:1px solid var(--line); border-radius:14px; padding:1.1rem 1.2rem; background:#fff;
  min-height:15rem;
}
.st-key-card_interaction h3, .st-key-card_alternatives h3{margin-top:0;}
.lead{color:var(--muted); font-size:1.1rem; margin-bottom:1.6rem;}
.note{color:var(--muted); font-size:.9rem;}
"""


@st.cache_resource
def _css() -> str:
    return "<style>" + font_face_css() + CSS.replace("__FALLBACK__", FALLBACK) + "</style>"


def inject_css() -> None:
    st.markdown(_css(), unsafe_allow_html=True)
