import os
import shutil

from logparser.Drain import LogParser
import pandas as pd

INPUT_LOG = "../data/Windows/Windows.log"
OUT_DIR   = "../data/Windows"
CSV_OUT   = "../data/Windows/Windows.log_structured.csv"

os.makedirs(OUT_DIR, exist_ok=True)

log_format = "<Date> <Time>, <Level> <Component> <Content>"

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
    st=0.5,
    depth=4,
)

parser.parse(os.path.basename(INPUT_LOG))

files = os.listdir(OUT_DIR)
structured_files = [f for f in files if f.endswith("_structured.csv")]

src = os.path.join(OUT_DIR, structured_files[0])

shutil.move(src, CSV_OUT)
print(f"Structured Windows log written to {CSV_OUT}")
