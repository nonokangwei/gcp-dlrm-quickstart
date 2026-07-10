"""Submit multi-node multi-GPU DLRM training to Vertex AI Custom Training."""
from google.cloud import aiplatform

PROJECT_ID = "project-kangwe-poc"
REGION = "us-central1"
STAGING_BUCKET = "gs://project-kangwe-poc-dlrm/staging"
IMAGE_URI = f"us-central1-docker.pkg.dev/{PROJECT_ID}/dlrm/dlrm-train:latest"

# 机器规格：按需改。n1-standard-16 + 4×T4 是和你本地 g4dn.12xlarge 对等的配置
MACHINE_TYPE = "n1-standard-16"
ACCELERATOR_TYPE = "NVIDIA_TESLA_T4"
ACCELERATOR_COUNT = 4
NUM_WORKER_NODES = 1  # 除 chief 外的 worker 节点数；总节点数 = 1 + NUM_WORKER_NODES

aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=STAGING_BUCKET)

pool_spec = {
    "machine_spec": {
        "machine_type": MACHINE_TYPE,
        "accelerator_type": ACCELERATOR_TYPE,
        "accelerator_count": ACCELERATOR_COUNT,
    },
    "replica_count": 1,
    "container_spec": {
        "image_uri": IMAGE_URI,
        "env": [{"name": "NCCL_DEBUG", "value": "INFO"}],
    },
}

worker_pool_specs = [
    pool_spec,                                        # workerpool0 = chief（必须 replica_count=1）
    {**pool_spec, "replica_count": NUM_WORKER_NODES}, # workerpool1 = workers
]
if NUM_WORKER_NODES == 0:
    worker_pool_specs = [pool_spec]

job = aiplatform.CustomJob(
    display_name="dlrm-multinode",
    worker_pool_specs=worker_pool_specs,
)
job.run(
    service_account=None,           # 需要能读写 gs://project-kangwe-poc-dlrm 的 SA
    enable_web_access=True,         # 可选：开交互式 shell 方便调试
)
