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


INPUT_LOG = "../data/Linux/Linux_utf8.log"
OUT_DIR   = "../data/Linux"

CSV_STRUCTURED_OUT = "../data/Linux/Linux.log_structured.csv"
CSV_TEMPLATES_OUT  = "../data/Linux/Linux.log_templates.csv"

os.makedirs(OUT_DIR, exist_ok=True)

log_format = "<Month> <Day> <Time> <Host> <Program>: <Content>"

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

files = os.listdir(OUT_DIR)

structured_files = [f for f in files if f.endswith("_structured.csv")]
template_files   = [f for f in files if f.endswith("_templates.csv")]

src_structured = os.path.join(OUT_DIR, structured_files[0])
src_templates  = os.path.join(OUT_DIR, template_files[0])

shutil.move(src_structured, CSV_STRUCTURED_OUT)
shutil.move(src_templates,  CSV_TEMPLATES_OUT)

print(f"Structured Linux log written to {CSV_STRUCTURED_OUT}")
print(f"Templates written to {CSV_TEMPLATES_OUT}")
