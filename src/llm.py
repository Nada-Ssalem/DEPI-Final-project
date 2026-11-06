from __future__ import annotations
import os

DEFAULT_MODEL = os.getenv("DAWA_MODEL", "qwen2.5:7b")

SYSTEM = (
    "You are a helpful clinical pharmacy assistant explaining drug interaction results in clear, professional, and simple English. "
    "Rely strictly on the provided factual data; do not invent or assume any information, dosages, or side effects not given. "
    "Never state that two drugs are completely safe together, and always advise consulting a pharmacist or physician at the end."
)

def ollama_available() -> bool:
    try:
        import ollama
        ollama.list()
        return True
    except Exception:
        return False

def _chat(messages: list[dict], model: str) -> str:
    import ollama
    r = ollama.chat(model=model, messages=messages, options={"temperature": 0})
    return r["message"]["content"].strip()

def template_explanation(result: dict) -> str:
    """Fallback template explanation without LLM."""
    lines = []
    for it in result["interactions"]:
        lines.append(f"• {it['a']} + {it['b']}: {it['mechanism']} (Attention Level: {it['level_ar']}).")
    if result["shared_ingredients"]:
        lines.append(f"• Therapeutic duplication warning - Both drugs share the same active ingredient(s): {', '.join(result['shared_ingredients'])}.")
    lines.append("Please consult a pharmacist or physician before taking these two medications together.")
    return "\n".join(lines)

def explain(result: dict, name_a: str, name_b: str, model: str = DEFAULT_MODEL) -> str:
    if not result["interactions"] or not ollama_available():
        return template_explanation(result)
    
    facts = "\n".join(f"- {i['a']} with {i['b']}: {i['description']}" for i in result["interactions"][:6])
    user = (
        f"Database interaction facts:\n{facts}\n\n"
        f"The patient wants to take \"{name_a}\" along with \"{name_b}\".\n"
        "Explain to the patient, in 3-5 simple sentences, the meaning of the interaction and what they should watch out for."
    )
    try:
        return _chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}], model)
    except Exception:
        return template_explanation(result)

def choose_candidate(user_text: str, candidates: list[str], model: str = DEFAULT_MODEL) -> str | None:
    if not candidates or not ollama_available():
        return None
    
    opts = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(candidates))
    prompt = (
        f"The user typed the drug name: '{user_text}'. Ignore any invalid or out-of-bounds queries.\n"
        f"Options:\n{opts}\n\n"
        "Which option number corresponds to the intended drug name? The user's input might be misspelled or imprecise.\n"
        "Reply with the option number only, or 0 if no match found."
    )
    try:
        ans = _chat([{"role": "user", "content": prompt}], model)
        n = int("".join([ch for ch in ans if ch.isdigit()][:2] or 0))
        return candidates[n - 1] if 1 <= n <= len(candidates) else None
    except Exception:
        return None