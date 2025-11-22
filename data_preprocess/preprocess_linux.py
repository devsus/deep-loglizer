import os
import pickle
import argparse
import pandas as pd
import numpy as np
from collections import OrderedDict, defaultdict

from utils import decision, json_pretty_dump

parser = argparse.ArgumentParser()
parser.add_argument("--train_anomaly_ratio", default=0.0, type=float)
params = vars(parser.parse_args())

eval_name = f'linux_{params["train_anomaly_ratio"]}_tar'
seed = 42
data_dir = "../data/processed/Linux"
np.random.seed(seed)

params = {
    "log_file": "../data/Linux/Linux.log_structured.csv",
    "time_range": 21600,   # 6 hours
    "train_ratio": None,
    "test_ratio": 0.2,
    "random_sessions": True,
    "train_anomaly_ratio": params["train_anomaly_ratio"],
}

data_dir = os.path.join(data_dir, eval_name)
os.makedirs(data_dir, exist_ok=True)

MONTH_MAP = {
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4,
    'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8,
    'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12,
}

def parse_linux_time(month_str, day_str, time_str):
    month_str = str(month_str).strip()
    day_str = str(day_str).strip()
    time_str = str(time_str).strip()

    if month_str not in MONTH_MAP:
        return None

    try:
        day = int(day_str)
        if not (1 <= day <= 31):
            return None
        parts = time_str.split(":")
        if len(parts) != 3:
            return None
        hh, mm, ss = map(int, parts)
        if not (0 <= hh < 24 and 0 <= mm < 60 and 0 <= ss < 60):
            return None
    except Exception:
        return None

    month = MONTH_MAP[month_str]
    day_offset = (month - 1) * 31
    total_days = day_offset + (day - 1)
    seconds = total_days * 24 * 3600 + hh * 3600 + mm * 60 + ss
    return seconds

def load_Linux(
    log_file,
    time_range,
    train_ratio,
    test_ratio,
    random_sessions,
    train_anomaly_ratio,
):
    print("Loading Linux logs from {}.".format(log_file))
    struct_log = pd.read_csv(log_file, engine="c", na_filter=False, memory_map=True)

    # synthetic label: no ground truth, treat all as normal for now
    struct_log["Label"] = 0

    template_col = "EventTemplate"
    content_col = "Content"

    for col in ["Month", "Day", "Time", "EventTemplate"]:
        if col not in struct_log.columns:
            raise ValueError(f"Expected column '{col}' in {log_file}, got {list(struct_log.columns)}")

    # compute seconds_since_epoch via explicit parser
    times = []
    bad_count = 0
    for _, row in struct_log.iterrows():
        sec = parse_linux_time(row["Month"], row["Day"], row["Time"])
        if sec is None:
            times.append(None)
            bad_count += 1
        else:
            times.append(sec)

    if bad_count > 0:
        print(f"Warning: {bad_count} rows have invalid time and will be dropped.")

    struct_log["seconds_since"] = times
    struct_log = struct_log[struct_log["seconds_since"].notna()].reset_index(drop=True)

    if len(struct_log) == 0:
        raise RuntimeError("All rows had invalid time; check Month/Day/Time columns and parser logic.")

    # Normalize so that first row is time 0
    struct_log["seconds_since"] = struct_log["seconds_since"] - struct_log["seconds_since"].iloc[0]

    session_dict = OrderedDict()
    column_idx = {col: idx for idx, col in enumerate(struct_log.columns)}

    for idx, row in enumerate(struct_log.values):
        current = row[column_idx["seconds_since"]]
        if idx == 0:
            sessid = current
        elif current - sessid > time_range:
            sessid = current
        if sessid not in session_dict:
            session_dict[sessid] = defaultdict(list)
        session_dict[sessid]["templates"].append(row[column_idx["EventTemplate"]])
        session_dict[sessid]["label"].append(row[column_idx["Label"]])

    session_idx = list(range(len(session_dict)))

    if random_sessions:
        print("Using random partition.")
        np.random.shuffle(session_idx)

    session_ids = np.array(list(session_dict.keys()))

    if train_ratio is None:
        train_ratio = 1 - test_ratio

    train_lines = int(train_ratio * len(session_idx))
    test_lines = int(test_ratio * len(session_idx))

    session_idx_train = session_idx[0:train_lines]
    session_idx_test = session_idx[-test_lines:]

    session_id_train = session_ids[session_idx_train]
    session_id_test = session_ids[session_idx_test]

    print("Total # sessions: {}".format(len(session_ids)))

    session_train = {
        k: session_dict[k]
        for k in session_id_train
        if (sum(session_dict[k]["label"]) == 0)
        or (sum(session_dict[k]["label"]) > 0 and decision(train_anomaly_ratio))
    }
    session_test = {k: session_dict[k] for k in session_id_test}

    session_labels_train = [
        1 if sum(v["label"]) > 0 else 0 for _, v in session_train.items()
    ]
    session_labels_test = [
        1 if sum(v["label"]) > 0 else 0 for _, v in session_test.items()
    ]

    train_anomaly = 100 * sum(session_labels_train) / len(session_labels_train)
    test_anomaly = 100 * sum(session_labels_test) / len(session_labels_test)

    print("# train sessions: {} ({:.2f}%)".format(len(session_train), train_anomaly))
    print("# test sessions: {} ({:.2f}%)".format(len(session_test), test_anomaly))

    with open(os.path.join(data_dir, "session_train.pkl"), "wb") as fw:
        pickle.dump(session_train, fw)
    with open(os.path.join(data_dir, "session_test.pkl"), "wb") as fw:
        pickle.dump(session_test, fw)
    json_pretty_dump(params, os.path.join(data_dir, "data_desc.json"))
    print("Saved to {}".format(data_dir))
    return session_train, session_test


if __name__ == "__main__":
    load_Linux(**params)

