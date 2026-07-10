#!/usr/bin/env python
"""
DLRM multi-GPU training (TorchRec + Mosaic StreamingDataset).

Launch with:
    torchrun --standalone --nproc_per_node=4 train_dlrm_multigpu.py
"""
import os
import sys
import shutil
import itertools
import dataclasses
from dataclasses import dataclass, field
from functools import partial
from typing import List, Optional
from datetime import datetime

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler
from torch.distributed._sharded_tensor import ShardedTensor

import torchmetrics as metrics
from tqdm import tqdm

from streaming import StreamingDataset, StreamingDataLoader
from streaming.base.util import clean_stale_shared_memory

from torchrec import EmbeddingBagCollection
from torchrec.distributed import TrainPipelineSparseDist
from torchrec.distributed.comm import get_local_size
from torchrec.distributed.model_parallel import DistributedModelParallel, get_default_sharders
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.distributed.planner.storage_reservations import HeuristicalStorageReservation
from torchrec.models.dlrm import DLRM, DLRMTrain
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.optim.keyed import CombinedOptimizer, KeyedOptimizerWrapper
from torchrec.optim.optimizers import in_backward_optimizer_filter
from torchrec.datasets.utils import Batch
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


# ----------------------------------------------------------------------------
# Config: dataset schema & paths (from notebook section 1.2 output)
# ----------------------------------------------------------------------------
CAT_DATA = {'cat_1': 103, 'cat_2': 180, 'cat_3': 93, 'cat_4': 15, 'cat_5': 107,
            'cat_6': 72, 'cat_7': 189, 'cat_8': 21, 'cat_9': 103, 'cat_10': 122}
DENSE_COLS = [f"int_{i}" for i in range(1, 11)]
CAT_COLS = [f"cat_{i}" for i in range(1, 11)]
EMB_COUNTS = [CAT_DATA[k] for k in CAT_COLS]

INPUT_DIR_TRAIN = "gs://project-kangwe-poc-dlrm/mds_data/train"
INPUT_DIR_VALIDATION = "gs://project-kangwe-poc-dlrm/mds_data/validation"
INPUT_DIR_TEST = "gs://project-kangwe-poc-dlrm/mds_data/test"
BUCKET_PATH = "gs://project-kangwe-poc-dlrm"
LOCAL_CACHE_ROOT = "/tmp/cache"

PROJECT_ID = "project-kangwe-poc"
REGION = "us-central1"
EXPERIMENT_NAME = "dlrm-training-demo"


# ----------------------------------------------------------------------------
# Args
# ----------------------------------------------------------------------------
@dataclass
class Args:
    epochs: int = 3
    embedding_dim: int = 128
    dense_arch_layer_sizes: list = field(default_factory=lambda: [512, 256, 128])
    over_arch_layer_sizes: list = field(default_factory=lambda: [512, 512, 256, 1])
    learning_rate: float = 0.03
    eps: float = 1e-8
    batch_size: int = 512
    print_sharding_plan: bool = True
    print_lr: bool = False
    lr_warmup_steps: int = 0
    lr_decay_start: int = 0
    lr_decay_steps: int = 0
    validation_freq: Optional[int] = None
    limit_train_batches: Optional[int] = None
    limit_val_batches: Optional[int] = None
    limit_test_batches: Optional[int] = None


@dataclass
class TrainValTestResults:
    val_aurocs: List[float] = field(default_factory=list)
    test_auroc: Optional[float] = None


def get_relevant_fields(args):
    keys = ["epochs", "embedding_dim", "dense_arch_layer_sizes",
            "over_arch_layer_sizes", "learning_rate", "eps", "batch_size"]
    result = {k: getattr(args, k) for k in keys}
    result["dense_cols"] = DENSE_COLS
    result["cat_cols"] = CAT_COLS
    result["emb_counts"] = EMB_COUNTS
    return result


def to_vertex_params(d):
    """Vertex AI log_params 只接受 int/float/bool/str；其余类型转成 str 以保留可读性。"""
    return {k: (v if isinstance(v, (int, float, bool, str)) else str(v)) for k, v in d.items()}


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def transform_to_torchrec_batch(batch, num_embeddings_per_feature: List[int]) -> Batch:
    cat_list = []
    for col_name in DENSE_COLS:
        val = batch[col_name].clone().detach().to(torch.float32)
        cat_list.append(val.unsqueeze(0).T)
    dense_features = torch.cat(cat_list, dim=1)

    kjt_values: List[int] = []
    kjt_lengths: List[int] = []
    for col_idx, col_name in enumerate(CAT_COLS):
        values = batch[col_name]
        for value in values:
            if value:
                kjt_values.append(int(value) % num_embeddings_per_feature[col_idx])
                kjt_lengths.append(1)
            else:
                kjt_lengths.append(0)

    sparse_features = KeyedJaggedTensor.from_lengths_sync(
        CAT_COLS,
        torch.tensor(kjt_values),
        torch.tensor(kjt_lengths, dtype=torch.int32),
    )
    labels = batch["label"].clone().detach().to(torch.int32)
    return Batch(dense_features=dense_features, sparse_features=sparse_features, labels=labels)


transform_partial = partial(transform_to_torchrec_batch, num_embeddings_per_feature=EMB_COUNTS)


def get_dataloader_with_mosaic(remote_path, local_cache_path, batch_size, label):
    print(f"[rank {os.environ.get('RANK', '?')}] Initializing {label} streaming dataset from: {remote_path}")
    dataset = StreamingDataset(
        remote=remote_path,
        local=local_cache_path,
        shuffle=True,
        batch_size=batch_size,
        download_timeout=300,
    )
    return StreamingDataLoader(dataset, batch_size=batch_size)


# ----------------------------------------------------------------------------
# LR scheduler
# ----------------------------------------------------------------------------
class LRPolicyScheduler(_LRScheduler):
    def __init__(self, optimizer, num_warmup_steps, decay_start_step, num_decay_steps):
        self.num_warmup_steps = num_warmup_steps
        self.decay_start_step = decay_start_step
        self.decay_end_step = decay_start_step + num_decay_steps
        self.num_decay_steps = num_decay_steps
        if self.decay_start_step < self.num_warmup_steps:
            sys.exit("Learning rate warmup must finish before the decay starts")
        super().__init__(optimizer)

    def get_lr(self):
        step_count = self._step_count
        if step_count < self.num_warmup_steps:
            scale = 1.0 - (self.num_warmup_steps - step_count) / self.num_warmup_steps
            lr = [base_lr * scale for base_lr in self.base_lrs]
            self.last_lr = lr
        elif self.decay_start_step <= step_count < self.decay_end_step:
            decayed_steps = step_count - self.decay_start_step
            scale = ((self.num_decay_steps - decayed_steps) / self.num_decay_steps) ** 2
            min_lr = 1e-7
            lr = [max(min_lr, base_lr * scale) for base_lr in self.base_lrs]
            self.last_lr = lr
        else:
            lr = self.last_lr if self.num_decay_steps > 0 else self.base_lrs
        return lr


# ----------------------------------------------------------------------------
# Checkpoint / metrics logging
# ----------------------------------------------------------------------------
def gather_and_get_state_dict(model):
    rank = dist.get_rank()
    state_dict = model.state_dict()
    gathered = {}
    for fqn, tensor in state_dict.items():
        if isinstance(tensor, ShardedTensor):
            full_tensor = torch.zeros(tensor.size()).to(tensor.device) if rank == 0 else None
            tensor.gather(0, full_tensor)
            if rank == 0:
                gathered[fqn] = full_tensor
        elif rank == 0:
            gathered[fqn] = tensor
    return gathered


def log_state_dict_to_gcs(model, epoch, bucket_path, run_name=None):
    state_dict = gather_and_get_state_dict(model)
    if dist.get_rank() != 0 or not state_dict:
        return None

    from google.cloud import storage, aiplatform

    local_path = f"/tmp/checkpoint_epoch_{epoch}.pt"
    torch.save(state_dict, local_path)
    bucket_name = bucket_path.split("/")[2]
    prefix = "/".join(bucket_path.split("/")[3:]).strip("/")
    subdir = f"models/{run_name}" if run_name else "models"
    blob_path = "/".join(p for p in [prefix, subdir, f"checkpoint_epoch_{epoch}.pt"] if p)
    storage.Client().bucket(bucket_name).blob(blob_path).upload_from_filename(local_path)
    os.remove(local_path)
    gcs_uri = f"gs://{bucket_name}/{blob_path}"
    print(f"Model saved to {gcs_uri}")

    # 把 checkpoint 作为 system.Model Artifact 记入 Vertex AI Metadata，并把 URI 写进当前 run 的 params
    try:
        aiplatform.Artifact.create(
            schema_title="system.Model",
            display_name=f"dlrm-ckpt-epoch-{epoch}" + (f"-{run_name}" if run_name else ""),
            uri=gcs_uri,
            metadata={"epoch": epoch, "framework": "pytorch-torchrec"},
        )
        aiplatform.log_params({f"checkpoint_epoch_{epoch}_uri": gcs_uri})
    except Exception as e:
        print(f"[warn] register model artifact failed: {e}")
    return gcs_uri


def log_metrics_gcp(metrics_dict, step=None):
    try:
        from google.cloud import aiplatform
        if step is not None:
            aiplatform.log_time_series_metrics(metrics_dict, step=step)
        else:
            aiplatform.log_metrics(metrics_dict)
    except Exception as e:
        print(f"[warn] vertex log failed: {e}")


# ----------------------------------------------------------------------------
# Train / eval loops
# ----------------------------------------------------------------------------
def batched(it, n):
    assert n >= 1
    for x in it:
        yield itertools.chain((x,), itertools.islice(it, n - 1))


def train_one_epoch(pipeline, train_dl, val_dl, epoch, lr_scheduler, args):
    pipeline._model.train()
    iterator = itertools.islice(iter(train_dl), args.limit_train_batches)
    is_rank_zero = dist.get_rank() == 0
    pbar = tqdm(iter(int, 1), desc=f"Epoch {epoch}", total=len(train_dl), disable=not is_rank_zero)

    start_it = 0
    n = args.validation_freq if args.validation_freq else len(train_dl)
    for batched_iterator in batched(iterator, n):
        for it in itertools.count(start_it):
            try:
                if is_rank_zero and args.print_lr:
                    for i, g in enumerate(pipeline._optimizer.param_groups):
                        print(f"lr: {it} {i} {g['lr']:.6f}")
                pipeline.progress(map(transform_partial, batched_iterator))
                lr_scheduler.step()
                pbar.update(1)
            except StopIteration:
                if is_rank_zero:
                    print(f"Total number of iterations: {it}")
                start_it = it
                break
        if args.validation_freq and start_it % args.validation_freq == 0:
            evaluate(args.limit_val_batches, pipeline, val_dl, "val")
            pipeline._model.train()
    pbar.close()


def evaluate(limit_batches, pipeline, eval_dl, stage):
    pipeline._model.eval()
    device = pipeline._device
    iterator = itertools.islice(iter(eval_dl), limit_batches)
    auroc = metrics.AUROC(task="binary").to(device)
    is_rank_zero = dist.get_rank() == 0
    pbar = tqdm(iter(int, 1), desc=f"Evaluating {stage} set", total=len(eval_dl), disable=not is_rank_zero)

    with torch.no_grad():
        while True:
            try:
                _loss, logits, labels = pipeline.progress(map(transform_partial, iterator))
                preds = torch.sigmoid(logits)
                auroc(preds, labels)
                pbar.update(1)
            except StopIteration:
                break
    pbar.close()

    auroc_result = auroc.compute().item()
    num_samples = torch.tensor(sum(map(len, auroc.target)), device=device)
    dist.reduce(num_samples, 0, op=dist.ReduceOp.SUM)
    if is_rank_zero:
        print(f"AUROC over {stage} set: {auroc_result}.")
        print(f"Number of {stage} samples: {num_samples}")
    return auroc_result


def train_val_test(args, model, optimizer, device, train_dl, val_dl, test_dl, lr_scheduler, run_name=None):
    results = TrainValTestResults()
    pipeline = TrainPipelineSparseDist(model, optimizer, device, execute_all_batches=True)
    is_rank_zero = dist.get_rank() == 0

    val_auroc = evaluate(args.limit_val_batches, pipeline, val_dl, "val")
    results.val_aurocs.append(val_auroc)
    if is_rank_zero:
        log_metrics_gcp({'val_auroc': val_auroc}, step=0)

    last_ckpt_uri = None
    for epoch in range(args.epochs):
        train_one_epoch(pipeline, train_dl, val_dl, epoch, lr_scheduler, args)
        val_auroc = evaluate(args.limit_val_batches, pipeline, val_dl, "val")
        results.val_aurocs.append(val_auroc)
        if is_rank_zero:
            log_metrics_gcp({'val_auroc': val_auroc}, step=epoch + 1)
        last_ckpt_uri = log_state_dict_to_gcs(pipeline._model.module, epoch, BUCKET_PATH, run_name=run_name)

    test_auroc = evaluate(args.limit_test_batches, pipeline, test_dl, "test")
    results.test_auroc = test_auroc
    if is_rank_zero:
        log_metrics_gcp({'test_auroc': test_auroc})
        if last_ckpt_uri:
            try:
                from google.cloud import aiplatform
                aiplatform.log_params({"final_model_uri": last_ckpt_uri})
            except Exception as e:
                print(f"[warn] log final_model_uri failed: {e}")
    return results


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main(args: Args):
    torch.jit._state.disable()
    global_rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    # 清理工作在 notebook 启动前完成；脚本内只做一次 barrier 保证同时进入数据加载
    print(f"[rank {global_rank}] process group ready", flush=True)
    dist.barrier()
    print(f"[rank {global_rank}] passed barrier, entering dataloader", flush=True)

    # 同一节点上所有 rank 共用同一个 local 目录：local leader (rank 0) 负责下载，其余 rank 等待并共享读取
    train_dl = get_dataloader_with_mosaic(INPUT_DIR_TRAIN, f"{LOCAL_CACHE_ROOT}/train", args.batch_size, "train")
    val_dl = get_dataloader_with_mosaic(INPUT_DIR_VALIDATION, f"{LOCAL_CACHE_ROOT}/val", args.batch_size, "val")
    test_dl = get_dataloader_with_mosaic(INPUT_DIR_TEST, f"{LOCAL_CACHE_ROOT}/test", args.batch_size, "test")

    # 数据加载完成后再启动 Vertex AI run（避免 rank 0 在 GCP 调用上阻塞导致其他 rank 等不到 index.json）
    run_name = f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    if global_rank == 0:
        try:
            from google.cloud import aiplatform
            aiplatform.init(project=PROJECT_ID, location=REGION, experiment=EXPERIMENT_NAME)
            aiplatform.start_run(run=run_name)aiplatform.log_params(to_vertex_params(get_relevant_fields(args)))
            
        except Exception as e:
            print(f"[warn] vertex init failed: {e}")

    eb_configs = [
        EmbeddingBagConfig(
            name=f"t_{feature_name}",
            embedding_dim=args.embedding_dim,
            num_embeddings=EMB_COUNTS[idx],
            feature_names=[feature_name],
        )
        for idx, feature_name in enumerate(CAT_COLS)
    ]
    dlrm_model = DLRM(
        embedding_bag_collection=EmbeddingBagCollection(tables=eb_configs, device=torch.device("meta")),
        dense_in_features=len(DENSE_COLS),
        dense_arch_layer_sizes=args.dense_arch_layer_sizes,
        over_arch_layer_sizes=args.over_arch_layer_sizes,
        dense_device=device,
    )
    train_model = DLRMTrain(dlrm_model)

    planner = EmbeddingShardingPlanner(
        topology=Topology(
            local_world_size=get_local_size(),
            world_size=dist.get_world_size(),
            compute_device=device.type,
        ),
        batch_size=args.batch_size,
        storage_reservation=HeuristicalStorageReservation(percentage=0.05),
    )
    plan = planner.collective_plan(train_model, get_default_sharders(), dist.GroupMember.WORLD)
    if global_rank == 0 and args.print_sharding_plan:
        print(plan)
    model = DistributedModelParallel(module=train_model, device=device, plan=plan)

    dense_optimizer = KeyedOptimizerWrapper(
        dict(in_backward_optimizer_filter(model.named_parameters())),
        lambda params: torch.optim.Adagrad(params, lr=args.learning_rate, eps=args.eps),
    )
    optimizer = CombinedOptimizer([model.fused_optimizer, dense_optimizer])
    lr_scheduler = LRPolicyScheduler(optimizer, args.lr_warmup_steps, args.lr_decay_start, args.lr_decay_steps)

    train_val_test(args, model, optimizer, device, train_dl, val_dl, test_dl, lr_scheduler, run_name=run_name)

    if global_rank == 0:
        try:
            from google.cloud import aiplatform
            aiplatform.end_run()
        except Exception:
            pass

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main(Args())
