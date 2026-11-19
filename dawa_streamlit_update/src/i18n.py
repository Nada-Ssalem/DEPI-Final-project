"""Arabic labels and small text helpers used by the Streamlit interface."""
from __future__ import annotations
import re

LEVEL_AR = {3: "عالية", 2: "متوسطة", 1: "منخفضة", 0: "—"}

# Keys are the labels produced by src/severity.py. A label that is not listed
# (for example one that is already Arabic in an older parquet file) is shown as is.
MECH_AR = {
    "QT prolongation": "إطالة فترة QT (خطر اضطراب نظم القلب)",
    "Bleeding risk": "زيادة خطر النزف",
    "Serotonin syndrome": "خطر متلازمة السيروتونين",
    "Respiratory depression": "تثبيط التنفس",
    "Arrhythmia": "اضطراب نظم القلب",
    "Nephrotoxicity / Cardiotoxicity / Neuromuscular block": "سمّية على الكلى أو القلب، أو حصار عصبي عضلي",
    "Neuromuscular block": "حصار عصبي عضلي",
    "Renal/Liver/Muscle toxicity": "سمّية على الكلى أو الكبد أو العضلات",
    "BP/Sugar changes": "تغيّر في ضغط الدم أو سكر الدم",
    "CNS Depression": "تثبيط الجهاز العصبي المركزي (نعاس)",
    "Hypotension": "هبوط ضغط الدم",
    "Hypoglycemia / Electrolyte imbalance": "هبوط السكر أو اضطراب الأملاح",
    "Drug accumulation (Toxicity risk)": "تراكم الدواء في الجسم (خطر التسمم)",
    "Decreased drug efficacy": "نقص فعالية الدواء",
    "Increased side effects": "زيادة الأعراض الجانبية",
    "Registered interaction (without automatic classification)": "تفاعل مسجّل (بدون تصنيف تلقائي)",
}

FORM_AR = {
    "tablet": "أقراص", "capsule": "كبسولات", "syrup": "شراب", "suspension": "معلّق",
    "injection": "حقن", "eye_drops": "قطرة للعين", "nasal": "للأنف", "drops": "نقط",
    "suppository": "لبوس", "cream": "كريم", "ointment": "مرهم", "gel": "جل",
    "lotion": "لوشن", "shampoo": "شامبو", "spray": "بخاخ", "inhaler": "جهاز استنشاق",
    "sachet": "أكياس", "solution": "محلول", "powder": "بودرة", "patch": "لصقة",
}

PACK_UNIT_AR = {"count": "وحدة", "ml": "مل", "g": "جم"}

REASON_AR = {
    "not_found": "الدواء غير موجود في قاعدة البيانات.",
    "no_strength": "التركيز غير مذكور في اسم الدواء، فلا يمكن التأكد من تطابق البديل.",
    "no_pack_or_form": "حجم العبوة أو الشكل الدوائي غير معروف.",
    "no_ingredients": "لا توجد مواد فعالة معروفة لهذا الدواء، فلا يمكن البحث عن بدائل.",
    "no_cheaper": "لا يوجد بديل أرخص بنفس التركيز وحجم العبوة.",
    "no_match_web": "لم نجد في قاعدة البيانات دواءً بنفس المواد الفعالة والتركيز والشكل الدوائي.",
}

_UNIT_TXT = {"mg": "mg", "mg/ml": "mg/ml", "iu": "IU", "pct": "%"}


def mech_ar(label: str) -> str:
    return MECH_AR.get(label, label)


def strength_to_text(keys) -> str:
    """('mg:400',) -> '400 mg'. The inverse of what src.parse.parse_name produces."""
    out = []
    for k in keys or ():
        unit, _, val = str(k).partition(":")
        out.append(f"{val} {_UNIT_TXT.get(unit, unit)}")
    return ", ".join(out)


def is_arabic(text: str) -> bool:
    return bool(re.search(r"[\u0600-\u06FF]", text or ""))
