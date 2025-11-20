import os
from logparser.Drain import LogParser
import pandas as pd

# Adjust these paths to your layout
INPUT_LOG = "../data/BGL/BGL.log"                 # full raw BGL log
OUT_DIR   = "../data/BGL/drain_output"            # Drain output dir
CSV_OUT   = "../data/BGL/BGL.log_structured_v2.csv"  # structured CSV for deep-loglizer

os.makedirs(OUT_DIR, exist_ok=True)

# 1) Drain configuration for BGL

# Raw line format:
# - 1117838570 2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.42.50.363779 R02-M1-N0-C:J12-U11 RAS KERNEL INFO instruction cache parity error corrected
log_format = "<Label> <Seq> <Date> <Node> <Time> <NodeRepeat> <Category> <Component> <Level> <Content>"

# Regex patterns to generalize variables (tune later if needed)
regex = [
    r"(\d+\.){3}\d+",                # IPv4 addresses (if any)
    r"\d+",                          # numbers (timestamps, IDs)
    r"0x[0-9A-Fa-f]+",               # hex codes
    r"R\d+-M\d+-N\d+-(C|I):J\d+-U\d+",  # BGL node ids
]

parser = LogParser(
    log_format=log_format,
    indir=os.path.dirname(INPUT_LOG),
    outdir=OUT_DIR,
    rex=regex,
    st=0.5,    # similarity threshold; 0.5 is a common starting point
    depth=4,   # tree depth; tune if clustering is bad
)

# 2) Run Drain on full BGL.log
parser.parse(os.path.basename(INPUT_LOG))

# 3) Drain outputs:
#    - <prefix>_structured.csv
#    - <prefix>_templates.csv
files = os.listdir(OUT_DIR)
structured_files = [f for f in files if f.endswith("_structured.csv")]
if not structured_files:
    raise RuntimeError(f"No *_structured.csv produced in {OUT_DIR}")

structured_path = os.path.join(OUT_DIR, structured_files[0])

# 4) Load, optionally normalize, and save to the path deep-loglizer expects
df = pd.read_csv(structured_path)

# Check columns: Drain usually gives something like:
# ["LineId","Label","Seq","Date","Node","Time","NodeRepeat","Category","Component","Level","Content","EventId","EventTemplate"]
# If columns look fine, just write:
df.to_csv(CSV_OUT, index=False)
print(f"Structured BGL log written to {CSV_OUT}")
