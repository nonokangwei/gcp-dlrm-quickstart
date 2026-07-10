# RedNote DLRM Demo — TorchRec on Google Cloud

一个端到端的 DLRM（Deep Learning Recommendation Model）推荐系统训练示例，演示如何在 **Google Cloud** 上完成从数据准备、单卡/多卡/多机训练、实验追踪到模型推理的完整流程。

本项目由 Databricks 官方 DLRM 示例迁移而来，把原本依赖 Unity Catalog / MLflow / TorchDistributor 的部分全部替换为 GCP 原生服务：

| Databricks 原版 | 本项目（GCP） |
|---|---|
| Unity Catalog Volumes | GCS + BigQuery |
| Databricks Runtime ML | Colab Enterprise / Vertex AI Workbench |
| MLflow Tracking | Vertex AI Experiments |
| MLflow Artifacts | GCS + Vertex AI Metadata (`system.Model` Artifact) |
| TorchDistributor（Spark） | `torchrun` + Vertex AI Custom Training |

---

## 目录结构

```
.
├── gcp-dlrm-quickstart.ipynb     主 notebook，端到端可执行（Step 1~7）
├── train_dlrm_multigpu.py      自包含的多卡/多机训练脚本（Step 5/6 使用）
├── launcher.py                 Vertex AI CLUSTER_SPEC → torchrun 启动器（Step 6）
├── Dockerfile                  训练容器镜像定义（Step 6）
├── submit_vertex_job.py        独立的作业提交脚本（可选，等价于 Step 6.5）
├── inference.py                从 Vertex AI Experiments 加载最新模型并推理（Step 7）
├── requirements.txt            提交端依赖（仅 google-cloud-aiplatform）
└── README.md
```

---

## 前置准备

### 1. GCP 项目与 API
在目标项目上启用以下 API：
- Vertex AI API
- Dataproc API（Serverless Spark，用于 Step 1 数据处理）
- BigQuery API
- Artifact Registry API
- Cloud Build API（Step 6 构建镜像用）
- Cloud Storage API

```bash
gcloud services enable \
    aiplatform.googleapis.com dataproc.googleapis.com bigquery.googleapis.com \
    artifactregistry.googleapis.com cloudbuild.googleapis.com storage.googleapis.com \
    --project=<YOUR_PROJECT_ID>
```

### 2. GCS 存储桶
需要两个桶（可复用同一个）：
- 数据/模型桶：存放 MDS 训练数据与 checkpoint，例如 `gs://<project>-dlrm`
- Staging 桶：Vertex AI 与 Cloud Build 的临时区，例如 `gs://<project>-experiments-staging-bucket`

### 3. 配额
Step 5/6 默认使用 `4 × NVIDIA T4`。请在 **IAM → 配额** 中确认所选 region 下：
- `Custom model training Nvidia T4 GPUs per region` ≥ 需要的总卡数（Step 6 双节点需 8）
- Colab Enterprise runtime 的 GPU 配额（Step 4/5 需要 4×T4 的 runtime）

### 4. 权限
执行 notebook 的身份 / Vertex AI Custom Job 的 service account 需要：
- `roles/aiplatform.user`
- `roles/storage.objectAdmin`（对上述两个桶）
- `roles/bigquery.dataEditor`
- `roles/artifactregistry.writer` + `roles/cloudbuild.builds.editor`（Step 6）

---

## 快速开始

1. 在 **Colab Enterprise** 中打开 `gcp-dlrm-quickstart.ipynb`
2. 选择带 **4 × T4 GPU** 的 runtime（Step 4/5 需要；只跑 Step 1 或 Step 6 可用 CPU runtime）
3. 修改 **Preparation** 段的 4 个参数（notebook 里是可交互的表单）：
   ```python
   PROJECT_ID      = "<your-project-id>"
   REGION          = "us-central1"
   BUCKET_URI      = "gs://<your-staging-bucket>"
   EXPERIMENT_NAME = "dlrm-training-demo"
   ```
4. 从上到下依次运行各 Step

> **提示**：Step 1（数据生成）只需跑一次；Step 4/5/6 是三种独立的训练方式，按需选择其一即可；Step 7 依赖 Step 4~6 中任意一次成功训练产生的 experiment run。

---

## Notebook 流程说明

### Step 1 — 数据准备（Dataproc Serverless Spark）
- 生成 10 万行合成点击数据（10 个连续特征 + 10 个类别特征 + 二分类标签）
- 写入 BigQuery 表 `recommender_systems.dlrm_sample_dataset`
- 用 Spark StandardScaler 归一化连续特征
- 7:2:1 切分后以 **MDS（Mosaic Data Shard）** 格式写入 `gs://<bucket>/mds_data/{train,validation,test}`

### Step 2 — 依赖安装与数据加载辅助函数
- 安装 `torch==2.2.2 / torchrec==0.6.0 / fbgemm-gpu==0.6.0`（**cu118** 版本）等
- **运行完安装 cell 后必须 Restart session** 再执行后续 cell，否则会遇到 ABI 不兼容的 import 错误
- 定义 `transform_to_torchrec_batch` / `get_dataloader_with_mosaic`

### Step 3 — 模型与训练逻辑
- `Args` 超参、`LRPolicyScheduler`、`train / evaluate / train_val_test`
- Vertex AI Experiments 集成：`log_params` / `log_time_series_metrics` / checkpoint 上传 GCS 并注册为 `system.Model` Artifact

### Step 4 — 单机单卡训练（notebook 内直接跑）
在当前 kernel 内设置 `WORLD_SIZE=1` 后直接调用 `main(args)`。用于快速验证代码正确性。

### Step 5 — 单机多卡训练（本地 4×GPU）
- 用 `%%writefile` 把训练逻辑落盘为 `vertex_build/train_dlrm_multigpu.py`
- 通过 `python -m torch.distributed.launch --nproc_per_node=4` 启动 4 个进程
- **不要在跑过 Step 4 之后不重启就直接跑 Step 5** —— 单卡训练会在 kernel 里残留 streaming 共享内存与端口占用，导致多卡进程死锁；先 Restart session

### Step 6 — 多机多卡训练（Vertex AI Custom Training）
| 子步骤 | 说明 |
|---|---|
| 6.1~6.3 | `%%writefile` 生成 `train_dlrm_multigpu.py` / `launcher.py` / `Dockerfile` 到 `vertex_build/` |
| 6.4 | `gcloud builds submit` 用 Cloud Build 构建并推送镜像到 Artifact Registry（无需本地 docker） |
| 6.5 | `aiplatform.CustomJob` 提交 chief + worker 双节点作业 |
| 6.6 | 查看作业状态、拉取 Cloud Logging 日志、在 Experiments 里查看指标曲线 |

调整 `NUM_WORKER_NODES` / `ACCELERATOR_TYPE` / `ACCELERATOR_COUNT` 即可扩缩容。

### Step 7 — 推理
- 从 Vertex AI Experiments 读取**最新 run** 的超参与 `final_model_uri`
- 从 GCS 下载 state_dict、重建 DLRM、加载权重
- 对 test 集抽样预测并对比标签

---

## 常见问题（踩过的坑）

| 现象 | 原因 | 解决 |
|---|---|---|
| `ImportError: cannot import name 'PreTrainedModel' from 'transformers'` | torchmetrics 1.0.3 与新版 transformers 5.x 不兼容 | 已在 Step 2.1 钉死 `transformers==4.44.2`；若不用文本指标也可直接卸载 transformers |
| `'_OpNamespace' 'fbgemm' object has no attribute 'jagged_2d_to_dense'` | fbgemm-gpu 从 PyPI 装到了 CUDA 12 版本，链接 `libcudart.so.12` 失败 | torch/torchrec/fbgemm-gpu **必须都从** `--index-url https://download.pytorch.org/whl/cu118` 安装 |
| `torchrun` 启动即 Segmentation fault | torch 2.2.x 的 c10d rendezvous 在 Python 3.12 上有 bug | notebook 里用 `python -m torch.distributed.launch` 代替；容器镜像用 Python 3.10 |
| Step 5 多卡进程 rank 1~3 报 `index.json took too long to download` | rank 0 被 kernel 残留的 streaming 共享内存/端口卡住，没人下载 index.json | 跑 Step 5 前 **Restart session**；启动 cell 已包含 `clean_stale_shared_memory()` + 清缓存 + 随机端口 |
| `aiplatform.log_params` 报 `Value ... is of type list` | Vertex AI 只接受 int/float/bool/str | 已用 `to_vertex_params()` 把 list 转成 str |
| Step 6 chief 节点 `master=localhost` 导致 worker 连不上 | Vertex AI 在 chief 的 `CLUSTER_SPEC` 里把自己写成 localhost | `launcher.py` 已用 `socket.getfqdn()` 替换，并改用独立端口 23456 |

---

## 自定义

- **换真实数据集（如 Criteo）**：改 Step 1 的数据读取逻辑，更新 Step 2.2 里的 `cat_data` / `dense_cols` / `cat_cols`
- **调超参**：改 Step 3.1 `Args`；Step 6 中可通过 `container_spec.args` 传参（需给训练脚本加 `argparse`）
- **换 GPU 型号**：Step 6.5 改 `MACHINE_TYPE="a2-highgpu-4g"` + `ACCELERATOR_TYPE="NVIDIA_TESLA_A100"`，同时把 Dockerfile 基础镜像与 `--index-url` 换成 cu121
- **注册到 Vertex AI Model Registry 并部署 Endpoint**：在 Step 7 之后追加 `aiplatform.Model.upload(artifact_uri=final_model_uri, serving_container_image_uri=...)`

---

## 参考
- [TorchRec](https://pytorch.org/torchrec/)
- [Facebook DLRM](https://github.com/facebookresearch/dlrm)
- [Mosaic StreamingDataset](https://docs.mosaicml.com/projects/streaming/)
- [Vertex AI Custom Training](https://cloud.google.com/vertex-ai/docs/training/custom-training)
- [Vertex AI Experiments](https://cloud.google.com/vertex-ai/docs/experiments/intro-vertex-ai-experiments)
