#!/usr/bin/env python
"""
Load the latest DLRM checkpoint from Vertex AI Experiments and run sample inference.

Equivalent of the original MLflow-based model loading + inference code, adapted to
the Vertex AI Experiments metadata (params + GCS checkpoint URI) produced by
train_dlrm_multigpu.py.
"""
import ast
import shutil

import torch
from google.cloud import aiplatform, storage

from streaming import StreamingDataset, StreamingDataLoader
from streaming.base.util import clean_stale_shared_memory

from torchrec import EmbeddingBagCollection
from torchrec.models.dlrm import DLRM
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
PROJECT_ID = "project-kangwe-poc"
REGION = "us-central1"
EXPERIMENT_NAME = "dlrm-training-demo"

TEST_DATA_REMOTE = "gs://project-kangwe-poc-dlrm/mds_data/test"
TEST_DATA_LOCAL = "/tmp/cache/infer_test"

aiplatform.init(project=PROJECT_ID, location=REGION, experiment=EXPERIMENT_NAME)


# ----------------------------------------------------------------------------
# Vertex AI Experiments helpers (replaces MLflow client calls)
# ----------------------------------------------------------------------------
def get_latest_run():
    """等价于 mlflow.search_runs(order_by=['start_time desc'], max_results=1)."""
    runs = aiplatform.ExperimentRun.list(experiment=EXPERIMENT_NAME)
    if not runs:
        raise RuntimeError(f"experiment {EXPERIMENT_NAME} 下没有任何 run")
    # run_name 形如 run-YYYYMMDD-HHMMSS，按名字倒序即最新
    return sorted(runs, key=lambda r: r.name, reverse=True)[0]


def get_latest_checkpoint_uri(params: dict) -> str:
    """等价于 get_latest_artifact_path：从 run params 里取最后一个 checkpoint 的 GCS URI."""
    if params.get("final_model_uri"):
        return params["final_model_uri"]
    ckpt_keys = sorted(
        k for k in params if k.startswith("checkpoint_epoch_") and k.endswith("_uri")
    )
    if not ckpt_keys:
        raise RuntimeError(
            "run params 里没有 checkpoint URI（final_model_uri / checkpoint_epoch_*_uri）"
        )
    return params[ckpt_keys[-1]]


def download_from_gcs(gcs_uri: str, local_path: str) -> str:
    bucket, blob = gcs_uri.replace("gs://", "").split("/", 1)
    storage.Client().bucket(bucket).blob(blob).download_to_filename(local_path)
    return local_path


def load_dlrm_from_vertex_run(run=None):
    """等价于 get_mlflow_model：从 Vertex AI run 的 params 重建 DLRM 并加载 state_dict."""
    device = torch.device("cuda")
    run = run or get_latest_run()
    print(f"Loading from experiment run: {run.name}")

    params = run.get_params()
    # list 类参数训练时用 str() 存的，用 ast.literal_eval 安全还原（不要用 eval）
    cat_cols = ast.literal_eval(params["cat_cols"])
    emb_counts = ast.literal_eval(params["emb_counts"])
    dense_cols = ast.literal_eval(params["dense_cols"])
    dense_arch_layer_sizes = ast.literal_eval(params["dense_arch_layer_sizes"])
    over_arch_layer_sizes = ast.literal_eval(params["over_arch_layer_sizes"])
    embedding_dim = int(params["embedding_dim"])

    ckpt_uri = get_latest_checkpoint_uri(params)
    print(f"Checkpoint: {ckpt_uri}")
    local_ckpt = download_from_gcs(ckpt_uri, "/tmp/dlrm_state_dict.pt")
    state_dict = torch.load(local_ckpt, map_location=device)
    # 训练时保存的是 DLRMTrain(...).module 的 state_dict，key 带 "model." 前缀 → 去掉
    state_dict = {
        (k[6:] if k.startswith("model.") else k): v for k, v in state_dict.items()
    }

    eb_configs = [
        EmbeddingBagConfig(
            name=f"t_{name}",
            embedding_dim=embedding_dim,
            num_embeddings=emb_counts[i],
            feature_names=[name],
        )
        for i, name in enumerate(cat_cols)
    ]
    dlrm_model = DLRM(
        embedding_bag_collection=EmbeddingBagCollection(tables=eb_configs, device=device),
        dense_in_features=len(dense_cols),
        dense_arch_layer_sizes=dense_arch_layer_sizes,
        over_arch_layer_sizes=over_arch_layer_sizes,
        dense_device=device,
    )
    dlrm_model.load_state_dict(state_dict)
    dlrm_model.to(device).eval()
    return dlrm_model, dense_cols, cat_cols, emb_counts


# ----------------------------------------------------------------------------
# Inference helpers
# ----------------------------------------------------------------------------
def get_dataloader_with_mosaic(remote_path, local_cache_path, batch_size, label):
    print(f"Initializing {label} streaming dataset from: {remote_path}")
    dataset = StreamingDataset(
        remote=remote_path,
        local=local_cache_path,
        shuffle=False,
        batch_size=batch_size,
    )
    return StreamingDataLoader(dataset, batch_size=batch_size)


def transform_test(batch, dense_cols, cat_cols, emb_counts):
    cat_list = [
        batch[c].clone().detach().to(torch.float32).unsqueeze(0).T for c in dense_cols
    ]
    dense_features = torch.cat(cat_list, dim=1)

    kjt_values, kjt_lengths = [], []
    for col_idx, col_name in enumerate(cat_cols):
        for value in batch[col_name]:
            if value:
                kjt_values.append(int(value) % emb_counts[col_idx])
                kjt_lengths.append(1)
            else:
                kjt_lengths.append(0)
    sparse_features = KeyedJaggedTensor.from_lengths_sync(
        cat_cols,
        torch.tensor(kjt_values),
        torch.tensor(kjt_lengths, dtype=torch.int32),
    )
    return dense_features, sparse_features


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    dlrm_model, dense_cols, cat_cols, emb_counts = load_dlrm_from_vertex_run()

    num_batches = 5
    batch_size = 1

    clean_stale_shared_memory()
    shutil.rmtree(TEST_DATA_LOCAL, ignore_errors=True)
    test_dataloader = iter(
        get_dataloader_with_mosaic(TEST_DATA_REMOTE, TEST_DATA_LOCAL, batch_size, "test")
    )

    device = torch.device("cuda:0")
    for _ in range(num_batches):
        next_batch = next(test_dataloader)
        expected_result = int(next_batch["label"][0])
        dense_features, sparse_features = transform_test(
            next_batch, dense_cols, cat_cols, emb_counts
        )
        with torch.no_grad():
            logits = dlrm_model(
                dense_features=dense_features.to(device),
                sparse_features=sparse_features.to(device),
            )
            prob = torch.sigmoid(logits)[0][0].item()
        print(
            f"Expected: {expected_result}; Predicted: {round(prob)} (prob={prob:.4f})"
        )
