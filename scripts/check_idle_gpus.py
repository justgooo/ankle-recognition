#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class GpuInfo:
    index: int
    name: str
    memory_used_mb: int
    memory_total_mb: int
    utilization_gpu: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail fast unless the selected nvidia-smi GPUs look idle."
    )
    parser.add_argument(
        "--gpu-ids",
        default="auto",
        help="Comma-separated nvidia-smi GPU indexes, or auto to inspect all visible GPUs.",
    )
    parser.add_argument("--require-count", type=int, default=0)
    parser.add_argument("--max-used-memory-mb", type=int, default=1024)
    parser.add_argument("--max-utilization", type=int, default=20)
    return parser.parse_args()


def parse_int(raw: str) -> int:
    return int(float(str(raw).strip().split()[0]))


def query_gpus() -> list[GpuInfo]:
    if shutil.which("nvidia-smi") is None:
        raise RuntimeError("nvidia-smi was not found")
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    gpus: list[GpuInfo] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        gpus.append(
            GpuInfo(
                index=parse_int(parts[0]),
                name=parts[1],
                memory_used_mb=parse_int(parts[2]),
                memory_total_mb=parse_int(parts[3]),
                utilization_gpu=parse_int(parts[4]),
            )
        )
    return gpus


def parse_gpu_ids(raw: str) -> list[int] | None:
    value = str(raw or "").strip()
    if not value or value.lower() == "auto":
        return None
    gpu_ids: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if token:
            gpu_ids.append(int(token))
    return gpu_ids


def main() -> int:
    args = parse_args()
    all_gpus = query_gpus()
    requested_ids = parse_gpu_ids(args.gpu_ids)
    if requested_ids is None:
        selected = all_gpus
    else:
        by_id = {gpu.index: gpu for gpu in all_gpus}
        missing = [gpu_id for gpu_id in requested_ids if gpu_id not in by_id]
        if missing:
            print(f"Missing requested GPU id(s): {missing}", file=sys.stderr)
            return 2
        selected = [by_id[gpu_id] for gpu_id in requested_ids]

    if args.require_count and len(selected) < args.require_count:
        print(
            f"Only {len(selected)} selected GPU(s), require {args.require_count}.",
            file=sys.stderr,
        )
        return 2

    busy = [
        gpu
        for gpu in selected
        if gpu.memory_used_mb > args.max_used_memory_mb
        or gpu.utilization_gpu > args.max_utilization
    ]
    print("Selected GPU idle check:")
    for gpu in selected:
        print(
            f"  GPU {gpu.index}: {gpu.name} | used={gpu.memory_used_mb} MiB/"
            f"{gpu.memory_total_mb} MiB | util={gpu.utilization_gpu}%"
        )
    if busy:
        print(
            "Refusing to start because at least one selected GPU exceeds "
            f"used<={args.max_used_memory_mb} MiB or util<={args.max_utilization}%.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
