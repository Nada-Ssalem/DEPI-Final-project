import re

HIGH, MODERATE, LOW = 3, 2, 1
LEVEL = {HIGH: "High", MODERATE: "Moderate", LOW: "Low"}

RULES = [
    (r"qtc|qt prolong", HIGH, "QT prolongation"),
    (r"increase the anticoagulant|increase the antiplatelet|bleeding|hemorrhag", HIGH, "Bleeding risk"),
    (r"serotonergic|serotonin", HIGH, "Serotonin syndrome"),
    (r"respiratory depress", HIGH, "Respiratory depression"),
    (r"arrhythmogenic|atrioventricular|bradycardic", HIGH, "Arrhythmia"),
    (r"increase the (nephrotoxic|cardiotoxic|hyperkalemic|neuromuscular)", HIGH, "Nephrotoxicity / Cardiotoxicity / Neuromuscular block"),
    (r"neuromuscular blocking", HIGH, "Neuromuscular block"),
    (r"risk or severity of (hyperkalemia|hypokalemia|hyponatremia|rhabdomyolysis|nephrotoxicity|hepatotoxicity|myopathy)", HIGH, "Renal/Liver/Muscle toxicity"),
    (r"risk or severity of (hypotension|hypertension|hypoglycemia|hyperglycemia)", MODERATE, "BP/Sugar changes"),
    (r"cns depressant|sedative|central nervous system depress", MODERATE, "CNS Depression"),
    (r"hypotensive|orthostatic", MODERATE, "Hypotension"),
    (r"hypoglycemic|hypokalemic|hypocalcemic", MODERATE, "Hypoglycemia / Electrolyte imbalance"),
    (r"serum concentration of .* can be increased|metabolism of .* can be decreased|decrease the excretion", MODERATE, "Drug accumulation (Toxicity risk)"),
    (r"efficacy|can be decreased|decrease the (absorption|antihypertensive|bioavailability)|serum concentration of .* (decreased|reduced)|metabolism of .* can", MODERATE, "Decreased drug efficacy"),
    (r"risk or severity of adverse effects", MODERATE, "Increased side effects"),
]
_COMPILED = [(re.compile(p), lvl, label) for p, lvl, label in RULES]


def rate(description: str) -> tuple[int, str]:
    d = str(description).lower()
    for rx, lvl, label in _COMPILED:
        if rx.search(d):
            return lvl, label
    return LOW, "Registered interaction (without automatic classification)"