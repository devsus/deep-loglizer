import os
import pickle
import argparse
import pandas as pd
import numpy as np
from utils import decision, json_pretty_dump
from collections import OrderedDict, defaultdict

parser = argparse.ArgumentParser()

parser.add_argument("--train_anomaly_ratio", default=0.0, type=float)

params = vars(parser.parse_args())

eval_name = f'windows_{params["--train_anomaly_ratio"]}_tar'
data_dir = "../data/processed/Windows"

seed = 42
np.random.seed(seed)

params ={
    "log_file": "../data/Windows/Windows_2k.log_structured.csv",
    "time_range": 21600,  # 6 hours (as BGL)
    "train_ratio": None,
    "test_ratio": 0.2,
    "random_sessions": True,
    "train_anomaly_ratio": params["train_anomaly_ratio"],
}

data_dir = os.path.join(data_dir, eval_name)
os.makedirs(data_dir, exist_ok=True)

def load_Windows(
    log_file,
    time_range,
    train_ratio,
    test_ratio,
    random_sessions,
    train_anomaly_ratio,
):
    print("Loading Windows logs from {}.".format(log_file))
    struct_log = pd.read_csv(log_file, engine="c", na_filter=False, memory_map=True)

    # assigning labels and processing timestamp
    # struct_log["Label"] = struct_log["Label"].map(lambda x: x != "-").astype(int).values
    struct_log["time"] = pd.to_datetime(
        struct_log["Date"] + struct_log["Time"], format="%Y-%m-%d-%H:%M:%S:%f"
    )
    struct_log["seconds_since"] = (
        (struct_log["time"] - struct_log["time"][0]).dt.total_seconds().astype(int)
    )

    session_dict = OrderedDict()
    column_idx = {col: idx for idx, col in enumerate(struct_log.columns)}
