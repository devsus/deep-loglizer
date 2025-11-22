import os
import shutil
from logparser.Drain import LogParser

INPUT_LOG = "../data/Apache/Apache.log"
OUT_DIR   = "../data/Apache"

CSV_STRUCTURED_OUT = "../data/Apache/Apache.log_structured.csv"
CSV_TEMPLATES_OUT  = "../data/Apache/Apache.log_templates.csv"

os.makedirs(OUT_DIR, exist_ok=True)

log_format = "<Content>"

regex = [
    r"(\d+\.){3}\d+",   # IPv4 addresses
    r"\d+",             # numbers
    r"0x[0-9A-Fa-f]+",  # hex values
    r"/[^\s]*",         # paths like /var/www/html/
]

parser = LogParser(
    log_format=log_format,
    indir=os.path.dirname(INPUT_LOG),
    outdir=OUT_DIR,
    rex=regex,
    st=0.5,
    depth=4,
)

print(f"Parsing file: {INPUT_LOG}")
parser.parse(os.path.basename(INPUT_LOG))

files = os.listdir(OUT_DIR)
structured_files = [f for f in files if f.endswith("_structured.csv")]
template_files   = [f for f in files if f.endswith("_templates.csv")]

if not structured_files:
    raise RuntimeError(f"No *_structured.csv produced in {OUT_DIR}")
if not template_files:
    raise RuntimeError(f"No *_templates.csv produced in {OUT_DIR}")

src_structured = os.path.join(OUT_DIR, structured_files[0])
src_templates  = os.path.join(OUT_DIR, template_files[0])

shutil.move(src_structured, CSV_STRUCTURED_OUT)
shutil.move(src_templates,  CSV_TEMPLATES_OUT)

print(f"Structured Apache log written to {CSV_STRUCTURED_OUT}")
print(f"Templates written to {CSV_TEMPLATES_OUT}")
