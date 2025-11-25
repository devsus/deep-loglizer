import os
import shutil
from logparser.Drain import LogParser

INPUT_LOG = "../data/Windows/Windows.log"
OUT_DIR   = "../data/Windows"

CSV_STRUCTURED   = "../data/Windows/Windows.log_structured.csv"
CSV_TEMPLATES  = "../data/Windows/Windows.log_templates.csv"

os.makedirs(OUT_DIR, exist_ok=True)

log_format = "<Date> <Time>, <Level> <Component> <Content>"

regex = [
    r"(\d+\.){3}\d+",            # IPv4
    r"\d+",                      # plain numbers
    r"0x[0-9A-Fa-f]+",           # hex codes
    r"[A-Za-z]:\\[^\s]*",        # paths like C:\Windows\...
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

src = os.path.join(OUT_DIR, structured_files[0])
src_templates  = os.path.join(OUT_DIR, template_files[0])

shutil.move(src, CSV_STRUCTURED)
shutil.move(src_templates, CSV_TEMPLATES)

print(f"Structured Windows log written to {CSV_STRUCTURED}")
print(f"Templates written to {CSV_TEMPLATES}")
