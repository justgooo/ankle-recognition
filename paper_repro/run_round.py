from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from paper_repro.common import load_manifest


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "paper_repro" / "papers.yaml"
LOG_DIR = REPO_ROOT / "paper_repro" / "logs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one paper reproduction round in parallel.")
    parser.add_argument("--round", required=True, dest="round_name", type=str)
    parser.add_argument("--gpu-ids", default="0,1,2", type=str)
    parser.add_argument("--timeout", default=10800, type=int)
    parser.add_argument("--append-root-results", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_manifest(MANIFEST_PATH)
    round_name = args.round_name
    if round_name not in manifest.get("rounds", {}):
        raise SystemExit(f"Unknown round: {round_name}")
    paper_ids = list(manifest["rounds"][round_name])
    gpu_ids = [gpu.strip() for gpu in args.gpu_ids.split(",") if gpu.strip()]
    if not gpu_ids:
        raise SystemExit("No GPU ids provided.")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Round={round_name} papers={paper_ids} gpu_ids={gpu_ids}")

    active: list[tuple[str, str, subprocess.Popen[bytes], Path]] = []
    completed_configs: list[str] = []

    def launch(paper_id: str, gpu_id: str) -> tuple[str, str, subprocess.Popen[bytes], Path]:
        config_path = REPO_ROOT / manifest["papers"][paper_id]["config"]
        log_path = LOG_DIR / f"{round_name}_{paper_id}_{timestamp}.log"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        command = [
            "timeout",
            str(args.timeout),
            sys.executable,
            "-m",
            "paper_repro.train",
            "--config",
            str(config_path),
        ]
        handle = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        process.log_handle = handle  # type: ignore[attr-defined]
        print(f"launch {paper_id} on cuda:{gpu_id} -> {log_path}")
        return paper_id, str(config_path), process, log_path

    queue = paper_ids[:]
    while queue or active:
        while queue and len(active) < len(gpu_ids):
            paper_id = queue.pop(0)
            gpu_id = gpu_ids[len(active)]
            active.append(launch(paper_id, gpu_id))

        next_active: list[tuple[str, str, subprocess.Popen[bytes], Path]] = []
        for paper_id, config_path, process, log_path in active:
            exit_code = process.poll()
            if exit_code is None:
                next_active.append((paper_id, config_path, process, log_path))
                continue
            handle = getattr(process, "log_handle", None)
            if handle is not None:
                handle.close()
            completed_configs.append(config_path)
            print(f"finish {paper_id} exit={exit_code} log={log_path}")
        if len(next_active) == len(active):
            first = active[0]
            exit_code = first[2].wait()
            handle = getattr(first[2], "log_handle", None)
            if handle is not None:
                handle.close()
            completed_configs.append(first[1])
            print(f"finish {first[0]} exit={exit_code} log={first[3]}")
            next_active = active[1:]
        active = next_active

    export_cmd = [
        sys.executable,
        "-m",
        "paper_repro.export_results",
        "--round",
        round_name,
    ]
    if args.append_root_results:
        export_cmd.append("--append-root-results")
    subprocess.run(export_cmd, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()

