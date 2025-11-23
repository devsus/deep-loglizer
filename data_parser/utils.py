from pathlib import Path

def convert_to_utf8(dataset_str):
    src = Path(f"../data/{dataset_str}/{dataset_str}.log")
    dst = Path(f"../data/{dataset_str}/{dataset_str}_utf8.log")
    with src.open("rb") as fin, dst.open("w", encoding="utf-8") as fout:
        for raw in fin:
            fout.write(raw.decode("latin1", errors="ignore"))
    print("Converted to UTF-8")
    return dst