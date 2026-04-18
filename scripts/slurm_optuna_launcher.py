from __future__ import annotations

import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
REMOTE_SENTINEL = "AUTORESEARCH_SLURM_REMOTE"


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def parse_job_list(raw_value: str) -> list[str]:
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def parse_group_list(raw_value: str, expected: int, name: str) -> list[str]:
    groups = [item.strip() for item in raw_value.split(";")]
    if len(groups) != expected:
        raise SystemExit(
            f"{name} must contain exactly {expected} ';'-separated group(s); got {len(groups)} from {raw_value!r}."
        )
    return groups


def resolve_repo_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def extract_flag_value(argv: list[str], flag: str) -> str | None:
    for index, token in enumerate(argv):
        if token == flag and index + 1 < len(argv):
            return argv[index + 1]
    return None


def load_study_root(search_config_path: Path) -> Path:
    with search_config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    raw_study_root = config.get("study", {}).get("study_root")
    if not raw_study_root:
        raise SystemExit(f"search config is missing study.study_root: {search_config_path}")
    return resolve_repo_path(str(raw_study_root))


def strip_conflicting_flags(argv: list[str]) -> list[str]:
    filtered: list[str] = []
    skip_next = False
    flags_with_values = {"--gpu-ids", "--max-workers"}
    flags_without_values = {"--sequential"}

    for token in argv:
        if skip_next:
            skip_next = False
            continue
        if token in flags_with_values:
            skip_next = True
            continue
        if token in flags_without_values:
            continue
        filtered.append(token)
    return filtered


def build_remote_shell(
    entrypoint: str,
    argv: list[str],
    gpu_ids: str,
    max_workers: int,
    resume: bool,
    skip_stale_running_cleanup: bool,
    extra_env: dict[str, str],
) -> str:
    forwarded = strip_conflicting_flags(argv)
    if resume and "--resume" not in forwarded:
        forwarded.append("--resume")
    if skip_stale_running_cleanup and "--skip-stale-running-cleanup" not in forwarded:
        forwarded.append("--skip-stale-running-cleanup")
    forwarded.extend(["--gpu-ids", gpu_ids, "--max-workers", str(max_workers)])

    env_prefix = {
        REMOTE_SENTINEL: "1",
        **{key: value for key, value in extra_env.items() if value},
    }
    env_text = " ".join(f"{key}={shlex.quote(value)}" for key, value in env_prefix.items())
    command = shell_join([sys.executable, entrypoint, *forwarded])
    if env_text:
        command = f"{env_text} {command}"
    return f"cd {shlex.quote(str(REPO_ROOT))} && {command}"


def wait_for_storage(storage_path: Path, leader: subprocess.Popen[str], timeout_seconds: int) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if storage_path.exists():
            return
        if leader.poll() is not None:
            break
        time.sleep(1)
    raise SystemExit(
        f"leader process did not create Optuna storage within {timeout_seconds}s: {storage_path}"
    )


def terminate_processes(processes: list[subprocess.Popen[str]]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        if process.poll() is None:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()


def maybe_dispatch_to_slurm_jobs(entrypoint: str, default_search_config: str) -> bool:
    if os.environ.get(REMOTE_SENTINEL) == "1":
        return False

    raw_job_ids = os.environ.get("AUTORESEARCH_SLURM_JOB_IDS", "").strip()
    if not raw_job_ids:
        return False

    job_ids = parse_job_list(raw_job_ids)
    if not job_ids:
        return False

    raw_gpu_groups = os.environ.get("AUTORESEARCH_SLURM_JOB_GPU_IDS", "").strip()
    if not raw_gpu_groups:
        raise SystemExit("AUTORESEARCH_SLURM_JOB_GPU_IDS is required when AUTORESEARCH_SLURM_JOB_IDS is set.")
    gpu_groups = parse_group_list(raw_gpu_groups, len(job_ids), "AUTORESEARCH_SLURM_JOB_GPU_IDS")

    raw_cuda_groups = os.environ.get("AUTORESEARCH_SLURM_JOB_CUDA_VISIBLE_DEVICES", "").strip()
    cuda_groups = (
        parse_group_list(raw_cuda_groups, len(job_ids), "AUTORESEARCH_SLURM_JOB_CUDA_VISIBLE_DEVICES")
        if raw_cuda_groups
        else [""] * len(job_ids)
    )

    wait_seconds = int(os.environ.get("AUTORESEARCH_SLURM_STORAGE_WAIT_SECONDS", "120"))
    search_config_raw = extract_flag_value(sys.argv[1:], "--search-config") or default_search_config
    search_config_path = resolve_repo_path(search_config_raw)
    study_root = load_study_root(search_config_path)
    storage_path = study_root / "study.sqlite3"

    launches: list[tuple[str, list[str]]] = []
    for index, job_id in enumerate(job_ids):
        gpu_group = gpu_groups[index]
        gpu_count = len([token for token in gpu_group.split(",") if token.strip()])
        if gpu_count <= 0:
            raise SystemExit(f"job {job_id} has an empty GPU group in AUTORESEARCH_SLURM_JOB_GPU_IDS.")
        extra_env = {}
        if cuda_groups[index]:
            extra_env["CUDA_VISIBLE_DEVICES"] = cuda_groups[index]
        remote_shell = build_remote_shell(
            entrypoint=entrypoint,
            argv=sys.argv[1:],
            gpu_ids=gpu_group,
            max_workers=gpu_count,
            resume=index > 0,
            skip_stale_running_cleanup=index > 0,
            extra_env=extra_env,
        )
        launches.append((job_id, ["srun", "--jobid", job_id, "--overlap", "bash", "-lc", remote_shell]))

    print("Dispatching Optuna study across Slurm jobs:")
    for index, (job_id, command) in enumerate(launches):
        role = "leader" if index == 0 else "follower"
        print(f"  - {role} job {job_id}: {shell_join(command)}", flush=True)

    processes: list[subprocess.Popen[str]] = []
    try:
        leader_job_id, leader_command = launches[0]
        print(f"Starting leader on job {leader_job_id}...", flush=True)
        leader = subprocess.Popen(leader_command, cwd=REPO_ROOT)
        processes.append(leader)

        if "--resume" not in sys.argv[1:]:
            wait_for_storage(storage_path, leader, timeout_seconds=wait_seconds)

        for follower_job_id, follower_command in launches[1:]:
            print(f"Starting follower on job {follower_job_id}...", flush=True)
            processes.append(subprocess.Popen(follower_command, cwd=REPO_ROOT))

        exit_codes = [process.wait() for process in processes]
    except BaseException:
        terminate_processes(processes)
        raise

    failed = [code for code in exit_codes if code != 0]
    if failed:
        raise SystemExit(f"distributed Slurm Optuna dispatch failed with exit codes: {exit_codes}")
    return True
