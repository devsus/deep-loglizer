import sys
import torch
import random
import os
import numpy as np
import json
import pickle
import random
import hashlib
import logging
from datetime import datetime

from deeploglizer.common.ddp import is_main_process


def dump_final_results(params, eval_results, model):
    result_str = "\t".join(["{}-{:.4f}".format(k, v) for k, v in eval_results.items()])

    key_info = [
        "dataset",
        "train_anomaly_ratio",
        "feature_type",
        "label_type",
        "use_attention",
    ]

    args = sys.argv
    model_name = args[0].replace("_demo.py", "")
    args = args[1:]
    input_params = [
        "{}:{}".format(args[idx * 2].strip("--"), args[idx * 2 + 1])
        for idx in range(len(args) // 2)
    ]
    recorded_params = ["{}:{}".format(k, v) for k, v in params.items() if k in key_info]

    params_str = "\t".join(input_params + recorded_params)

    with open(os.path.join(f"{params['dataset']}.txt"), "a+") as fw:
        info = "{} {} {} {} {} train: {:.3f} test: {:.3f}\n".format(
            datetime.now().strftime("%Y%m%d-%H%M%S"),
            params["hash_id"],
            model_name,
            params_str,
            result_str,
            model.time_tracker["train"],
            model.time_tracker["test"],
        )
        fw.write(info)

    #aggregated runtime metrics
    try:
        save_dir = os.path.join("./experiment_records", params["hash_id"])
        os.makedirs(save_dir, exist_ok=True)

        metrics = {}

        epoch_times = model.time_tracker.get("train_epoch_times", [])
        epoch_times_arr = np.array(epoch_times)
        metrics["train_epochs"] = epoch_times_arr.size
        metrics["train_epoch_time_mean_sec"] = epoch_times_arr.mean()
        metrics["train_epoch_time_p95_sec"] = np.percentile(epoch_times_arr, 95)

        epoch_throughput = model.time_tracker.get("train_epoch_throughput", [])
        thr_arr = np.array(epoch_throughput)
        metrics["train_throughput_mean_samples_sec"] = thr_arr.mean()
        metrics["train_throughput_mean_p95_samples_sec"] = np.percentile(thr_arr, 95)

        train_total = model.time_tracker.get("train_total")
        metrics["train_total_time_sec"] = train_total

        max_alloc = model.time_tracker.get("gpu_train_max_memory_allocated_bytes")
        metrics["gpu_train_max_memory_allocated_mb"] = max_alloc / (1024.0 ** 2)

        max_reserved = model.time_tracker.get("gpu_train_max_memory_reserved_bytes")
        metrics["gpu_train_max_memory_reserved_mb"] = max_reserved / (1024.0 ** 2)

        util_samples = model.time_tracker.get("gpu_train_util_samples", [])
        util_arr = np.asarray(util_samples)
        metrics["gpu_train_util_mean_pct"] = util_arr.mean() if util_arr.size > 0 else None
        metrics["gpu_train_util_p95_pct"] = np.percentile(util_arr, 95) if util_arr.size > 0 else None

        mem_samples = model.time_tracker.get("gpu_train_mem_samples_bytes", [])
        mem_arr = np.asarray(mem_samples)
        metrics["gpu_train_mem_mean_mb"] = mem_arr.mean() / (1024 ** 2) if mem_arr.size > 0 else None
        metrics["gpu_train_mem_p95_mb"] = (np.percentile(mem_arr, 95) / (1024 ** 2)) if mem_arr.size > 0 else None

        metrics["world_size"] = model.world_size

        metrics_path = os.path.join(save_dir, "metrics.json")
        json_pretty_dump(metrics, metrics_path)
    except Exception as e:
        logging.warning(f"Failed to dump metrics: {e}")

# quick fix for duplicate stdout
def dump_params(params):
    hash_id = params.get("hash_id") #!
    if not hash_id:
        hash_id = hashlib.md5(
            str(sorted([(k, v) for k, v in params.items()])).encode("utf-8")
        ).hexdigest()[0:8]
        params["hash_id"] = hash_id

    save_dir = os.path.join("./experiment_records", hash_id)
    os.makedirs(save_dir, exist_ok=True)

    if is_main_process():
        json_pretty_dump(params, os.path.join(save_dir, "params.json"))

        log_file = os.path.join(save_dir, hash_id + ".log")
        # logs will not show in the file without the two lines.
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s P%(process)d %(levelname)s %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )

        logging.info(json.dumps(params, indent=4))
    else:
        log_file = os.path.join(save_dir, hash_id + ".log")
        # logs will not show in the file without the two lines.
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s P%(process)d %(levelname)s %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )
    return save_dir


def decision(probability):
    return random.random() < probability


def json_pretty_dump(obj, filename):
    with open(filename, "w") as fw:
        json.dump(
            obj,
            fw,
            sort_keys=True,
            indent=4,
            separators=(",", ": "),
            ensure_ascii=False,
        )


def tensor2flatten_arr(tensor):
    return tensor.data.cpu().numpy().reshape(-1)


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def set_device(gpu=-1):
    if gpu != -1 and torch.cuda.is_available():
        device = torch.device("cuda:" + str(gpu))
    else:
        device = torch.device("cpu")
    return device


def dump_pickle(obj, file_path):
    logging.info("Dumping to {}".format(file_path))
    with open(file_path, "wb") as fw:
        pickle.dump(obj, fw)


def load_pickle(file_path):
    logging.info("Loading from {}".format(file_path))
    with open(file_path, "rb") as fr:
        return pickle.load(fr)

