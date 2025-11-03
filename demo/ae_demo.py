#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import shutil
import sys

from deeploglizer.common.ddp import setup, is_main_process, cleanup

sys.path.append("../")
import argparse
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist

from deeploglizer.models import AutoEncoder
from deeploglizer.common.preprocess import FeatureExtractor
from deeploglizer.common.dataloader import load_sessions, log_dataset
from deeploglizer.common.utils import seed_everything, dump_params, dump_final_results


parser = argparse.ArgumentParser()

##### Model params
parser.add_argument("--model_name", default="Autoencoder", type=str)
parser.add_argument("--hidden_size", default=128, type=int)
parser.add_argument("--num_directions", default=2, type=int)
parser.add_argument("--num_layers", default=2, type=int)
parser.add_argument("--embedding_dim", default=32, type=int)

##### Dataset params
parser.add_argument("--dataset", default="HDFS", type=str)
parser.add_argument(
    "--data_dir", default="../data/processed/HDFS_100k/hdfs_0.0_tar", type=str
)
parser.add_argument("--window_size", default=10, type=int)
parser.add_argument("--stride", default=1, type=int)

##### Input params
parser.add_argument("--feature_type", default="sequentials", type=str, choices=["sequentials", "semantics"])
parser.add_argument("--use_tfidf", action="store_true")
parser.add_argument("--max_token_len", default=50, type=int)
parser.add_argument("--min_token_count", default=1, type=int)
# Uncomment the following to use pretrained word embeddings. The "embedding_dim" should be set as 300
# parser.add_argument(
#     "--pretrain_path", default="../data/pretrain/wiki-news-300d-1M.vec", type=str
# )

##### Training params
parser.add_argument("--epoches", default=100, type=int)
parser.add_argument("--batch_size", default=1024, type=int)
parser.add_argument("--learning_rate", default=0.01, type=float)
parser.add_argument("--anomaly_ratio", default=0.1, type=float)
parser.add_argument("--patience", default=3, type=int)

##### Others
parser.add_argument("--random_seed", default=42, type=int)
parser.add_argument("--gpu", default=0, type=int)
# parser.add_argument("--cache", default=False, type=bool)  # trap?
parser.add_argument("--cache", action="store_true")

params = vars(parser.parse_args())

#model_save_path = dump_params(params)


if __name__ == "__main__":
    is_ddp, local_rank = setup()

    model_save_path = dump_params(params)

    seed_everything(params["random_seed"])

    session_train, session_test = load_sessions(data_dir=params["data_dir"])
    ext = FeatureExtractor(**params)

    # cache handling for DDP; same as LSTM
    # try main process instead of local rank
    if params["cache"] and (not is_ddp or local_rank == 0):
        shutil.rmtree(getattr(ext, "cache_dir", "./cache"), ignore_errors=True)
        os.makedirs(ext.cache_dir, exist_ok=True)

    if is_ddp:
        dist.barrier(device_ids=[local_rank])  # !

    if params["cache"] and is_ddp:
        if is_main_process():
            ext.fit(session_train)
            session_train = ext.transform(session_dict=session_train, datatype="train")
            session_test = ext.transform(session_dict=session_test, datatype="test")
        dist.barrier()
        if not is_main_process():
            assert ext.load(), f"Rank {dist.get_rank()} failed to load cached feature extractor."
            session_train = ext.transform(session_dict=session_train, datatype="train")
            session_test = ext.transform(session_dict=session_test, datatype="test")
    else:
        session_train = ext.fit_transform(session_train)
        session_test = ext.transform(session_test, datatype="test")

    #session_train = ext.fit_transform(session_train)
    #session_test = ext.transform(session_test, datatype="test")

    dataset_train = log_dataset(session_train, feature_type=params["feature_type"])
    train_sampler = DistributedSampler(dataset_train, shuffle=True, drop_last=False) if is_ddp else None
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=params["batch_size"],
        shuffle=True,
        pin_memory=True,
        num_workers=3,
        persistent_workers=True,
        prefetch_factor=4,
    )

    dataset_test = log_dataset(session_test, feature_type=params["feature_type"])
    dataloader_test = DataLoader(
        dataset_test,
        batch_size=4096,
        shuffle=False,
        pin_memory=True,
        num_workers=1,
        persistent_workers=False,
    )

    model = AutoEncoder(
        meta_data=ext.meta_data, model_save_path=model_save_path, **params
    )

    eval_results = model.fit(
        dataloader_train,
        test_loader=dataloader_test,
        epoches=params["epoches"],
        learning_rate=params["learning_rate"],
    )

    if is_main_process():
        print(eval_results)
        dump_final_results(params, eval_results, model)     # compare this to stdout in LSTM

    # clean cache
    if params["cache"] and is_ddp:
        shutil.rmtree(getattr(ext, "cache_dir", "./cache"), ignore_errors=True)

    # destroy DDP process group
    try:
        del dataloader_train
        del dataloader_test
    except NameError:
        pass
    if dist.is_initialized():
        dist.barrier()
    cleanup()