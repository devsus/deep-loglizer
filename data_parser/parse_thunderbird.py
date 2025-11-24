import os
import shutil
from logparser.Drain import LogParser

input_log = "../data/Thunderbird/Thunderbird.log"
out_dir = "../data/Thunderbird"

csv_structured = "../data/Thunderbird/Thunderbird.log_structured.csv"
csv_templates = "../data/Thunderbird/Thunderbird.log_templates.csv"

os.makedirs(out_dir, exist_ok=True)

log_format = (
    "<Label> <Timestamp> <Date> <Node> "
    "<Month> <Day> <Time> <NodeRepeat> "
    "<Component> <Content>"
)

rex = [
    r'((25[0-5]|2[0-4]\d|[0-1]?\d?\d)(\.(?!$)|$)){4}',  # IP
    r'\d+', # plain numbers
]

parser = LogParser(
    log_format=log_format,
    indir=input_log,
    outdir=out_dir,
    rex=rex,
    st=0.5,
    depth=5,
)

parser.parse(os.path.basename(input_log))

files = os.listdir(out_dir)

structured_files = [f for f in files if f.endswith("_structured.csv")]
template_files   = [f for f in files if f.endswith("_templates.csv")]

src = os.path.join(out_dir, structured_files[0])
src_templates  = os.path.join(out_dir, template_files[0])

shutil.move(src, csv_structured)
shutil.move(src_templates, csv_templates)

print(f"Structured Windows log written to {csv_structured}")
print(f"Templates written to {csv_templates}")
