import json
from os import getenv
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

PROJECT_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(PROJECT_ROOT / ".env")

MODEL_NAME = "gemini-3.5-flash-lite"

PROMPT = """
You are a pharmaceutical product identification assistant.

Your task is to identify a medicine from its commercial name
and/or an uploaded package image.

Extract only information that can be reasonably identified.

Important rules:
1. Never invent an active ingredient or a price.
2. Do not confuse the brand name with the active ingredient.
3. Read the package image carefully if provided.
4. Extract strength, dosage form, and approximate price (in EGP if Egyptian market or stated currency) only when visible or reasonably known; otherwise use null.
5. Prices are approximate and may vary; never treat them as guaranteed or official prices.
6. If the medicine identity is ambiguous, return unknown.
7. If the image is unclear, return unknown.
8. Do not provide medical advice or dosage recommendations.
9. Do not guess based only on a similar-sounding brand name.

Return valid JSON using exactly this structure:

{
    "status": "identified or unknown",
    "brand_name": null,
    "active_ingredients": [],
    "strength": [],
    "dosage_form": null,
    "approximate_price": null,
    "confidence": 0.0,
    "reason": ""
}

The confidence must be a number between 0 and 1.

Use status "identified" only when you can identify
the medicine with reasonable confidence.

If uncertain, use status "unknown" and leave
active_ingredients empty.
"""


def unknown_result(reason="Medicine could not be identified."):
    return {
        "status": "unknown",
        "brand_name": None,
        "active_ingredients": [],
        "strength": [],
        "dosage_form": None,
        "approximate_price": None,
        "confidence": 0.0,
        "reason": reason,
    }


def _clean_price(value):
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned if cleaned else None
    return None


def validate_result(data):
    if not isinstance(data, dict):
        return unknown_result("Invalid response format.")

    status = data.get("status")

    if status != "identified":
        return unknown_result(
            str(data.get("reason") or "Medicine identity is uncertain.")
        )

    ingredients = data.get("active_ingredients")

    if not isinstance(ingredients, list):
        return unknown_result("Invalid active ingredients.")

    ingredients = [
        item.strip()
        for item in ingredients
        if isinstance(item, str) and item.strip()
    ]

    try:
        confidence = float(data.get("confidence", 0))
    except (TypeError, ValueError):
        confidence = 0.0

    confidence = max(0.0, min(1.0, confidence))

    if not ingredients or confidence < 0.80:
        return unknown_result(
            "The medicine identity is not sufficiently certain."
        )

    strength = data.get("strength", [])

    if not isinstance(strength, list):
        strength = []

    strength = [
        item.strip()
        for item in strength
        if isinstance(item, str) and item.strip()
    ]

    brand_name = data.get("brand_name")
    dosage_form = data.get("dosage_form")
    approximate_price = _clean_price(data.get("approximate_price"))

    return {
        "status": "identified",
        "brand_name": (
            brand_name.strip()
            if isinstance(brand_name, str) and brand_name.strip()
            else None
        ),
        "active_ingredients": ingredients,
        "strength": strength,
        "dosage_form": (
            dosage_form.strip()
            if isinstance(dosage_form, str) and dosage_form.strip()
            else None
        ),
        "approximate_price": approximate_price,
        "confidence": confidence,
        "reason": str(data.get("reason") or ""),
    }


def identify_medicine(
    medicine_name=None,
    image_bytes=None,
    mime_type=None,
):
    """
    Identify a medicine using its name, package image, or both.

    Returns:
        dict with status:
        - identified
        - unknown
        - error
    """
    api_key = getenv("GEMINI_API_KEY")

    if not api_key:
        return {
            "status": "error",
            "reason": "GEMINI_API_KEY is missing.",
        }

    if not medicine_name and not image_bytes:
        return unknown_result("No medicine name or image provided.")

    contents = [PROMPT]

    if medicine_name:
        contents.append(
            f"Medicine commercial name: {medicine_name}"
        )

    if image_bytes:
        allowed_types = {
            "image/jpeg",
            "image/png",
            "image/webp",
            "image/heic",
            "image/heif",
        }

        if mime_type not in allowed_types:
            return unknown_result("Unsupported image format.")

        contents.append(
            types.Part.from_bytes(
                data=image_bytes,
                mime_type=mime_type,
            )
        )

    try:
        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
            ),
        )

        if not response.text:
            return unknown_result("Gemini returned an empty response.")

        data = json.loads(response.text)

        return validate_result(data)

    except json.JSONDecodeError:
        return unknown_result("Gemini returned invalid JSON.")

    except Exception as exc:
        return {
            "status": "error",
            "reason": f"Gemini API error: {exc}",
        }