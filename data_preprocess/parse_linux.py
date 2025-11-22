import os
import shutil
from pathlib import Path

from logparser.Drain import LogParser



src = Path("../data/Linux/Linux.log")
dst = Path("../data/Linux/Linux_utf8.log")
with src.open("rb") as fin, dst.open("w", encoding="utf-8") as fout:
    for raw in fin:
        fout.write(raw.decode("latin1", errors="ignore"))
print("Converted to UTF-8")


INPUT_LOG = "../data/Linux/Linux_utf8.log"              # raw .log file
OUT_DIR   = "../data/Linux"      # where Drain writes CSVs

# Final structured + templates CSVs you will feed into preprocess_linux.py
CSV_STRUCTURED_OUT = "../data/Linux/Linux.log_structured.csv"
CSV_TEMPLATES_OUT  = "../data/Linux/Linux.log_templates.csv"

os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# Drain configuration for Linux syslog format
# ----------------------------------------------------------------------
# Example raw line:
# Jun  9 06:06:20 combo kernel: Mount-cache hash table entries: 512 (order: 0, 4096 bytes)
#
# Pattern:
# <Month> <Day> <Time> <Host> <Program>: <Content>

log_format = "<Month> <Day> <Time> <Host> <Program>: <Content>"

# Regexes to generalize variables into <*>
regex = [
    r"(\d+\.){3}\d+",            # IPv4 addresses
    r"\d+",                      # numbers (PIDs, counters, ports, etc.)
    r"0x[0-9A-Fa-f]+",           # hex values
]

parser = LogParser(
    log_format=log_format,
    indir=os.path.dirname(INPUT_LOG),
    outdir=OUT_DIR,
    rex=regex,
    st=0.5,   # similarity threshold
    depth=4,  # tree depth
)

parser.parse(os.path.basename(INPUT_LOG))

# ----------------------------------------------------------------------
# Move Drain's structured + templates CSV into final locations
# ----------------------------------------------------------------------
files = os.listdir(OUT_DIR)

structured_files = [f for f in files if f.endswith("_structured.csv")]
template_files   = [f for f in files if f.endswith("_templates.csv")]

if not structured_files:
    raise RuntimeError(f"No *_structured.csv produced in {OUT_DIR}")
if not template_files:
    raise RuntimeError(f"No *_templates.csv produced in {OUT_DIR}")

src_structured = os.path.join(OUT_DIR, structured_files[0])
src_templates  = os.path.join(OUT_DIR, template_files[0])

# move instead of reading into memory
shutil.move(src_structured, CSV_STRUCTURED_OUT)
shutil.move(src_templates,  CSV_TEMPLATES_OUT)

print(f"Structured Linux log written to {CSV_STRUCTURED_OUT}")
print(f"Templates written to {CSV_TEMPLATES_OUT}")
