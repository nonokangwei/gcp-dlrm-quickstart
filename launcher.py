#!/usr/bin/env python
"""Vertex AI Custom Training launcher: parse CLUSTER_SPEC -> torchrun multi-node."""
import json
import os
import socket
import sys
import shutil
import subprocess

from streaming.base.util import clean_stale_shared_memory

TORCHRUN_PORT = 23456  # 独立于 Vertex AI 的 CLUSTER_SPEC 端口，避免与其健康检查冲突


def parse_cluster_spec():
    spec = os.environ.get("CLUSTER_SPEC")
    if not spec:
        return {"master_addr": "127.0.0.1", "master_port": TORCHRUN_PORT,
                "nnodes": 1, "node_rank": 0}

    spec = json.loads(spec)
    cluster = spec["cluster"]
    task = spec["task"]

    pools = sorted(cluster.keys())
    hosts = []
    for p in pools:
        hosts.extend(cluster[p])
    nnodes = len(hosts)

    my_pool = task["type"]
    my_index = int(task["index"])
    node_rank = 0
    for p in pools:
        if p == my_pool:
            node_rank += my_index
            break
        node_rank += len(cluster[p])

    master_host = hosts[0].split(":")[0]
    # Vertex AI 在 chief 节点上会把自己写成 localhost，替换为可被其他节点解析的主机名
    if master_host in ("localhost", "127.0.0.1"):
        master_host = socket.getfqdn() if node_rank == 0 else master_host
    return {"master_addr": master_host, "master_port": TORCHRUN_PORT,
            "nnodes": nnodes, "node_rank": node_rank}


def main():
    cfg = parse_cluster_spec()
    nproc = os.environ.get("NPROC_PER_NODE")
    if not nproc:
        import torch
        nproc = torch.cuda.device_count()

    clean_stale_shared_memory()
    shutil.rmtree("/tmp/cache", ignore_errors=True)

    cmd = [
        "torchrun",
        f"--nnodes={cfg['nnodes']}",
        f"--node_rank={cfg['node_rank']}",
        f"--nproc_per_node={nproc}",
        f"--master_addr={cfg['master_addr']}",
        f"--master_port={cfg['master_port']}",
        "train_dlrm_multigpu.py",
    ] + sys.argv[1:]

    print(f"[launcher] node_rank={cfg['node_rank']}/{cfg['nnodes']} "
          f"master={cfg['master_addr']}:{cfg['master_port']} nproc={nproc}", flush=True)
    print(f"[launcher] exec: {' '.join(cmd)}", flush=True)
    os.environ["PYTHONUNBUFFERED"] = "1"
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
