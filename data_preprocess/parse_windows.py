import os
from logparser.Drain import LogParser
import pandas as pd

# ----------------------------------------------------------------------
# Paths — ADJUST these for your setup
# ----------------------------------------------------------------------
# Raw full Windows log from LogHub (the big one, not the 2k sample)
INPUT_LOG = "../data/Windows/Windows.log"          # e.g. CBS.log or whatever you named it

# Directory where Drain will drop *_structured.csv and *_templates.csv
OUT_DIR   = "../data/Windows"

# Final structured CSV you will feed into preprocess_windows.py
CSV_OUT   = "../data/Windows/Windows.log_structured.csv"

os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# Drain configuration
# ----------------------------------------------------------------------
# Raw line shape (CBS example):
# 2016-09-28 04:30:30, Info                  CBS    Starting TrustedInstaller initialization.
#
# -> we model as:
# <Date> <Time>, <Level> <Component> <Content>

log_format = "<Date> <Time>, <Level> <Component> <Content>"

# Regexes to generalize variables into <*>
# Tune later if needed; start conservative.
regex = [
    r"(\d+\.){3}\d+",            # IPv4
    r"\d+",                      # plain numbers
    r"0x[0-9A-Fa-f]+",           # hex codes / HRESULTs
    r"[A-Za-z]:\\[^\s]*",        # Windows paths like C:\Windows\...
]

parser = LogParser(
    log_format=log_format,
    indir=os.path.dirname(INPUT_LOG),
    outdir=OUT_DIR,
    rex=regex,
    st=0.5,   # similarity threshold
    depth=4,  # tree depth
)

# ----------------------------------------------------------------------
# Run Drain
# ----------------------------------------------------------------------
parser.parse(os.path.basename(INPUT_LOG))

# Drain writes two files in OUT_DIR:
#   <prefix>_structured.csv
#   <prefix>_templates.csv
files = os.listdir(OUT_DIR)
structured_files = [f for f in files if f.endswith("_structured.csv")]
if not structured_files:
    raise RuntimeError(f"No *_structured.csv produced in {OUT_DIR}")

structured_path = os.path.join(OUT_DIR, structured_files[0])

# ----------------------------------------------------------------------
# Normalize / copy structured CSV to where deep-loglizer expects it
# ----------------------------------------------------------------------
df = pd.read_csv(structured_path)

# Expect columns like:
# ["LineId","Date","Time","Level","Component","Content","EventId","EventTemplate","ParameterList"]
# That is perfectly fine for preprocess_windows.py as long as Date, Time, EventTemplate exist.

df.to_csv(CSV_OUT, index=False)
print(f"Structured Windows log written to {CSV_OUT}")
