import re
import pandas as pd

tpl = pd.read_csv("../data/Windows/Windows_2k.log_templates.csv", engine="c", na_filter=False, memory_map=True)

# mark likely failures/errors as 1, others 0 (GPT)
POS = {
    r"\berror\b": 5,
    r"\bfail(?:ed|ure|ing)?\b": 4,
    r"\bexception\b": 5,
    r"\btimeout\b": 4,
    r"\bfatal\b": 5,
    r"\bcrash(ed)?\b": 5,
    r"\bpanic\b": 5,
    r"\binvalid\b": 3,
    r"\bcorrupt(?:ion)?\b": 5,
    r"\baccess\s+denied\b": 5,
    r"\bpermission\s+denied\b": 5,
    r"\bnot\s+found\b": 3,
    r"\bmissing\b": 3,
    r"\bunavailable\b": 3,
    r"\brefused\b": 3,
    r"\bviolation\b": 4,
    r"\babort(?:ed)?\b": 4,
    r"\bassert(?:ion)?\b": 4,
    r"\brollback\b": 3,
    # Windows-style codes
    r"\bHRESULT\b": 5,
    r"\b0x8[0-9A-Fa-f]{7}\b": 5,  # many error-class HRESULTs
    r"\b0xC[0-9A-Fa-f]{7}\b": 5,  # NTSTATUS failures
}

NEG = {
    r"\binfo\b": -3,
    r"\bdebug\b": -2,
    r"\bstarting\b": -1,
    r"\binitialized?\b": -1,
    r"\bsuccess(?:ful|fully)?\b": -3,
    r"\bok\b": -2,
    r"\bqueued\b": -1,
}

POS_RE = [(re.compile(p, re.I), v) for p, v in POS.items()]
NEG_RE = [(re.compile(p, re.I), v) for p, v in NEG.items()]

def score_template(text: str) -> int:
    score = 0
    for rx, v in POS_RE:
        if rx.search(text):
            score += v
    for rx, v in NEG_RE:
        if rx.search(text):
            score += v
    return score

THRESHOLD = 4

labels = []
for t in tpl["EventTemplate"].astype(str):
    s = score_template(t)
    labels.append(1 if s >= THRESHOLD else 0)

out = tpl.copy()
out["Label"] = labels

out = out[["EventId", "EventTemplate", "Label"]]

out.to_csv("../data/Windows/anomaly_label.csv", index=False)
print(f"wrote ../data/Windows/anomaly_label.csv with {len(out)} templates")
print(f"anomalous templates: {out['Label'].sum()} ({100*out['Label'].mean():.2f}%)")
