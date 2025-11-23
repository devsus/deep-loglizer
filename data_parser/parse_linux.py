import os
import shutil
from logparser.Drain import LogParser
from utils import convert_to_utf8

INPUT_LOG = convert_to_utf8("Linux")
OUT_DIR   = "../data/Linux"

CSV_STRUCTURED = "../data/Linux/Linux.log_structured.csv"
CSV_TEMPLATES  = "../data/Linux/Linux.log_templates.csv"

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
    st=0.5,
    depth=4,
)

parser.parse(os.path.basename(INPUT_LOG))

files = os.listdir(OUT_DIR)

structured_files = [f for f in files if f.endswith("_structured.csv")]
template_files   = [f for f in files if f.endswith("_templates.csv")]

src_structured = os.path.join(OUT_DIR, structured_files[0])
src_templates  = os.path.join(OUT_DIR, template_files[0])

shutil.move(src_structured, CSV_STRUCTURED)
shutil.move(src_templates, CSV_TEMPLATES)

print(f"Structured Linux log written to {CSV_STRUCTURED}")
print(f"Templates written to {CSV_TEMPLATES}")
