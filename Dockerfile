FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip git ca-certificates && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
    pip install "numpy<2.0.0" && \
    pip install \
        torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 \
        torchrec==0.6.0 fbgemm-gpu==0.6.0 \
        --index-url https://download.pytorch.org/whl/cu118 && \
    pip install \
        torchmetrics==1.0.3 \
        iopath==0.1.10 \
        pyre_extensions==0.0.32 \
        mosaicml-streaming==0.7.5 \
        google-cloud-aiplatform \
        google-cloud-storage \
        tqdm

WORKDIR /app
COPY train_dlrm_multigpu.py launcher.py /app/

ENTRYPOINT ["python", "launcher.py"]
