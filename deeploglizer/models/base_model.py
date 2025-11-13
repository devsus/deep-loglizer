import os
import time
import torch
import torch.distributed as dist
import logging
import math
import numpy as np
import pandas as pd
from torch import nn
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

from deeploglizer.common.utils import set_device, tensor2flatten_arr
from deeploglizer.common.ddp import is_main_process

class Embedder(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        pretrain_matrix=None,
        freeze=False,
        use_tfidf=False,
    ):
        super(Embedder, self).__init__()
        self.use_tfidf = use_tfidf
        if pretrain_matrix is not None:
            self.embedding_layer = nn.Embedding.from_pretrained(
                pretrain_matrix, padding_idx=1, freeze=freeze
            )
        else:
            self.embedding_layer = nn.Embedding(
                vocab_size, embedding_dim, padding_idx=1
            )

    def forward(self, x):
        if self.use_tfidf:
            return torch.matmul(x, self.embedding_layer.weight.double())
        else:
            return self.embedding_layer(x.long())


class ForcastBasedModel(nn.Module):
    def __init__(
        self,
        meta_data,
        model_save_path,
        feature_type,
        label_type,
        eval_type,
        topk,
        use_tfidf,
        embedding_dim,
        freeze=False,
        gpu=-1,
        multi_gpu=False, #!
        anomaly_ratio=None,
        patience=3,
        **kwargs,
    ):
        super(ForcastBasedModel, self).__init__()
        self.device = set_device(gpu)
        self.multi_gpu = multi_gpu #!
        self.topk = topk
        self.meta_data = meta_data
        self.feature_type = feature_type
        self.label_type = label_type
        self.eval_type = eval_type
        self.anomaly_ratio = anomaly_ratio  # only used for auto encoder
        self.patience = patience
        self.time_tracker = {}

        os.makedirs(model_save_path, exist_ok=True)
        self.model_save_file = os.path.join(model_save_path, "model.ckpt")
        if feature_type in ["sequentials", "semantics"]:
            self.embedder = Embedder(
                meta_data["vocab_size"],
                embedding_dim=embedding_dim,
                pretrain_matrix=meta_data.get("pretrain_matrix", None),
                freeze=freeze,
                use_tfidf=use_tfidf,
            )
        else:
            logging.info(f'Unrecognized feature type, expect sequentials or semantics, got {feature_type}')

    def evaluate(self, test_loader, dtype="test"):
        logging.info("Evaluating {} data.".format(dtype))

        if self.label_type == "next_log":
            return self.__evaluate_next_log(test_loader, dtype=dtype)
        elif self.label_type == "anomaly":
            return self.__evaluate_anomaly(test_loader, dtype=dtype)
        elif self.label_type == "none":
            return self.__evaluate_recst(test_loader, dtype=dtype)

    def __evaluate_recst(self, test_loader, dtype="test"):
        self.eval()  # set to evaluation mode
        with torch.no_grad():
            y_pred = []
            store_dict = defaultdict(list)
            infer_start = time.time()
            for batch_input in test_loader:
                return_dict = self.forward(self.__input2device(batch_input))
                y_pred = return_dict["y_pred"]
                store_dict["session_idx"].extend(
                    tensor2flatten_arr(batch_input["session_idx"])
                )
                store_dict["window_anomalies"].extend(
                    tensor2flatten_arr(batch_input["window_anomalies"])
                )
                store_dict["window_preds"].extend(tensor2flatten_arr(y_pred))
            infer_end = time.time()
            logging.info("Finish inference [{:.2f}s]".format(infer_end - infer_start))
            self.time_tracker["test"] = infer_end - infer_start

            store_df = pd.DataFrame(store_dict)

            use_cols = ["session_idx", "window_anomalies", "window_preds"]
            session_df = (
                store_df[use_cols]
                .groupby("session_idx", as_index=False)
                .max()  # most anomalous window
            )
            assert (
                self.anomaly_ratio is not None
            ), "anomaly_ratio should be specified for autoencoder!"
            thre = np.percentile(
                session_df[f"window_preds"].values, 100 - self.anomaly_ratio * 100
            )
            pred = (session_df[f"window_preds"] > thre).astype(int)
            y = (session_df["window_anomalies"] > 0).astype(int)

            eval_results = {
                "f1": f1_score(y, pred),
                "rc": recall_score(y, pred),
                "pc": precision_score(y, pred),
                "acc": accuracy_score(y, pred),
            }
            logging.info({k: f"{v:.3f}" for k, v in eval_results.items()})
            return eval_results

    def __evaluate_anomaly(self, test_loader, dtype="test"):

        self.eval()  # set to evaluation mode
        with torch.no_grad():
            y_pred = []
            store_dict = defaultdict(list)
            infer_start = time.time()
            for batch_input in test_loader:
                return_dict = self.forward(self.__input2device(batch_input))
                y_prob, y_pred = return_dict["y_pred"].max(dim=1)
                store_dict["session_idx"].extend(
                    tensor2flatten_arr(batch_input["session_idx"])
                )
                store_dict["window_anomalies"].extend(
                    tensor2flatten_arr(batch_input["window_anomalies"])
                )
                store_dict["window_preds"].extend(tensor2flatten_arr(y_pred))
            infer_end = time.time()
            logging.info("Finish inference. [{:.2f}s]".format(infer_end - infer_start))
            self.time_tracker["test"] = infer_end - infer_start

            store_df = pd.DataFrame(store_dict)
            use_cols = ["session_idx", "window_anomalies", "window_preds"]
            session_df = store_df[use_cols].groupby("session_idx", as_index=False).sum()
            pred = (session_df[f"window_preds"] > 0).astype(int)
            y = (session_df["window_anomalies"] > 0).astype(int)

            eval_results = {
                "f1": f1_score(y, pred),
                "rc": recall_score(y, pred),
                "pc": precision_score(y, pred),
                "acc": accuracy_score(y, pred),
            }
            logging.info({k: f"{v:.3f}" for k, v in eval_results.items()})
            return eval_results

    def __evaluate_next_log(self, test_loader, dtype="test"):
        model = self.eval()  # set to evaluation mode
        with torch.no_grad():
            y_pred = []
            store_dict = defaultdict(list)
            infer_start = time.time()
            for batch_input in test_loader:
                return_dict = model.forward(self.__input2device(batch_input))
                y_pred = return_dict["y_pred"]
                y_prob_topk, y_pred_topk = torch.topk(y_pred, self.topk)  # b x topk

                store_dict["session_idx"].extend(
                    tensor2flatten_arr(batch_input["session_idx"])
                )
                store_dict["window_anomalies"].extend(
                    tensor2flatten_arr(batch_input["window_anomalies"])
                )
                store_dict["window_labels"].extend(
                    tensor2flatten_arr(batch_input["window_labels"])
                )
                store_dict["x"].extend(batch_input["features"].data.cpu().numpy())
                store_dict["y_pred_topk"].extend(y_pred_topk.data.cpu().numpy())
                store_dict["y_prob_topk"].extend(y_prob_topk.data.cpu().numpy())
            infer_end = time.time()
            logging.info("Finish inference. [{:.2f}s]".format(infer_end - infer_start))
            self.time_tracker["test"] = infer_end - infer_start
            store_df = pd.DataFrame(store_dict)
            best_result = None
            best_f1 = -float("inf")

            count_start = time.time()

            """topkdf = pd.DataFrame(store_df["y_pred_topk"].tolist())
            logging.info("Calculating acc sum.")
            hit_df = pd.DataFrame()
            for col in sorted(topkdf.columns):
                topk = col + 1
                hit = (topkdf[col] == store_df["window_labels"]).astype(int)
                hit_df[topk] = hit
                if col == 0:
                    acc_sum = 2 ** topk * hit
                else:
                    acc_sum += 2 ** topk * hit
            acc_sum[acc_sum == 0] = 2 ** (1 + len(topkdf.columns))
            hit_df["acc_num"] = acc_sum

            for col in sorted(topkdf.columns):
                topk = col + 1
                check_num = 2 ** topk
                store_df["window_pred_anomaly_{}".format(topk)] = (
                    ~(hit_df["acc_num"] <= check_num)
                ).astype(int)
            # store_df.to_csv("store_{}_2.csv".format(dtype), index=False)"""

            # True top-k accuracy - ChatGPT 5
            topkdf = pd.DataFrame(store_df["y_pred_topk"].tolist())
            logging.info("Calculating acc sum.")
            # bool matrix: prediction at rank j equals true label
            eq_mat = topkdf.eq(store_df["window_labels"].to_numpy(), axis=0)

            # build window_pred_anomaly_k and cache top-k accuracies
            topk_acc = {}
            for col in sorted(topkdf.columns):
                topk = col + 1
                # hit if true label appears anywhere within first k predictions
                hit_k = eq_mat.iloc[:, :topk].any(axis=1)
                store_df[f"window_pred_anomaly_{topk}"] = (~hit_k).astype(int)
                topk_acc[topk] = float(hit_k.mean())
            # store_df.to_csv("store_{}_2.csv".format(dtype), index=False)

            logging.info("Finish generating store_df.")

            if self.eval_type == "session":
                use_cols = ["session_idx", "window_anomalies"] + [
                    f"window_pred_anomaly_{topk}" for topk in range(1, self.topk + 1)
                ]
                session_df = (
                    store_df[use_cols].groupby("session_idx", as_index=False).sum()
                )
            else:
                session_df = store_df
            # session_df.to_csv("session_{}_2.csv".format(dtype), index=False)

            for topk in range(1, self.topk + 1):
                pred = (session_df[f"window_pred_anomaly_{topk}"] > 0).astype(int)
                y = (session_df["window_anomalies"] > 0).astype(int)

                #window_topk_acc = 1 - store_df["window_anomalies"].sum() / len(store_df)
                #window_topk_acc = float(hit_df[topk].mean())
                window_topk_acc = topk_acc[topk]    # !

                eval_results = {
                    "f1": f1_score(y, pred),
                    "rc": recall_score(y, pred),
                    "pc": precision_score(y, pred),
                    "top{}-acc".format(topk): float(window_topk_acc),
                }
                logging.info({k: f"{v:.3f}" for k, v in eval_results.items()})
                if eval_results["f1"] >= best_f1:
                    best_result = eval_results
                    best_f1 = eval_results["f1"]
            count_end = time.time()
            logging.info("Finish counting [{:.2f}s]".format(count_end - count_start))
            return best_result

    def __input2device(self, batch_input):
        return {k: v.to(self.device) for k, v in batch_input.items()}

    def save_model(self):
        logging.info("Saving model to {}".format(self.model_save_file))
        try:
            torch.save(
                self.state_dict(),
                self.model_save_file,
                _use_new_zipfile_serialization=False,
            )
        except:
            torch.save(self.state_dict(), self.model_save_file)

    def load_model(self, model_save_file=""):
        logging.info("Loading model from {}".format(self.model_save_file))
        self.load_state_dict(torch.load(model_save_file, map_location=self.device, weights_only=True)) # !

    def fit(
            self,
            train_loader,
            test_loader=None,
            epoches=10,
            learning_rate=1.0e-3,
            lr_scheduler=None,
            lr_target=None,
            warmup_steps=0):    #!
        # detect DDP and set the correct CUDA device
        is_ddp = dist.is_initialized()
        local_rank = int(os.environ["LOCAL_RANK"]) if is_ddp else 0
        self.device = torch.device(f'cuda:{local_rank}') if torch.cuda.is_available() else torch.device("cpu")

        self.to(self.device)

        # wrap once
        model = self
        if is_ddp:
            from torch.nn.parallel import DistributedDataParallel as DDP
            model = DDP(self, device_ids=[local_rank], output_device=local_rank)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # !

        # ---- LR schedule: warmup to lr_target, then cosine back to base ----
        base_lr = float(learning_rate)
        target_lr = float(lr_target) if lr_target is not None else base_lr
        total_steps = max(1, epoches * len(train_loader))
        warmup_steps = int(warmup_steps)

        def lr_at_step(step_idx: int) -> float:
            # step_idx is 0-based; we set LR *before* optimizer.step()
            if lr_scheduler is None or target_lr == base_lr:
                return base_lr
            if step_idx < warmup_steps:
                # linear warmup from base_lr -> target_lr
                alpha = (step_idx + 1) / max(1, warmup_steps)
                return base_lr + (target_lr - base_lr) * alpha
            # cosine decay from target_lr -> base_lr across the remaining steps
            remain = max(1, total_steps - warmup_steps)
            progress = min(1.0, (step_idx - warmup_steps) / remain)
            return base_lr + 0.5 * (target_lr - base_lr) * (1.0 + math.cos(math.pi * progress))

        # --------------------------------------------------------------------

        logging.info(
            "Start training on {} batches with {}.".format(
                len(train_loader), self.device
            )
        )
        best_f1 = -float("inf")
        best_results = None
        worse_count = 0
        for epoch in range(1, epoches + 1):
            epoch_time_start = time.time()

            if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)

            #model = self.train()
            #model = self.__wrap_model()
            model.train()
            # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            batch_cnt = 0
            epoch_loss_num = 0.0  # sum off loss * batch_size across all ranks
            sample_count = 0.0 # total samples across all ranks
            global_step = 0
            for batch_input in train_loader:
                # set per-step LR
                current_lr = lr_at_step(global_step)    #!
                for g in optimizer.param_groups:
                    g["lr"] = current_lr

                loss = model.forward(self.__input2device(batch_input))["loss"]
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1    #!

                # global reduction for logging
                # batch size (first tensor in batch; all share leading dim)
                first_tensor = next(iter(batch_input.values()))
                batch_size = int(first_tensor.size(0))

                # per-batch num = mean_loss * batch_size
                loss_num = (loss.detach() * batch_size).to(self.device)
                count = torch.tensor([batch_size], dtype=loss_num.dtype, device=self.device)

                if dist.is_initialized():
                    dist.all_reduce(loss_num, op=dist.ReduceOp.SUM)
                    dist.all_reduce(count, op=dist.ReduceOp.SUM)

                epoch_loss_num += loss_num.item()
                sample_count += count.item()

                # epoch_loss += loss.item()
                batch_cnt += 1
            epoch_loss = epoch_loss_num / max(sample_count, 1) #!
            epoch_time_elapsed = time.time() - epoch_time_start
            if is_main_process(): #!
                logging.info(
                    "Epoch {}/{}, training loss: {:.5f} [{:.2f}s], lr {:.6f}".format(
                        epoch, epoches, epoch_loss, epoch_time_elapsed, current_lr)
                )   #!
            self.time_tracker["train"] = epoch_time_elapsed

            if dist.is_initialized(): #!
                dist.barrier()

            stop = False #!
            if test_loader is not None and (epoch % 1 == 0): #!
                if is_main_process(): # rank 0 only
                    eval_results = self.evaluate(test_loader)
                    if eval_results["f1"] > best_f1:
                        best_f1 = eval_results["f1"]
                        best_results = eval_results
                        best_results["converge"] = int(epoch)
                        self.save_model()
                        worse_count = 0
                    else:
                        worse_count += 1
                        if worse_count >= self.patience:
                            logging.info("Early stop at epoch: {}".format(epoch))
                            #break
                            stop = True
            if dist.is_initialized(): # broadcast early stopping to all ranks
                flag = torch.tensor([1 if stop else 0], device=self.device)
                if dist.get_rank() == 0: # only value from rank 0 matters, broadcast it
                    pass
                dist.broadcast(flag, src=0)
                stop = bool(flag.item())
            if stop:
                break

        if is_main_process():
            self.load_model(self.model_save_file)
        return best_results
