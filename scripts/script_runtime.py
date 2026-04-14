from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PYTHON_CANDIDATES = [
    ".venv/bin/python",
    ".venv/Scripts/python.exe",
    sys.executable,
    "python3",
    "python",
]


def repo_relative(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def tail_lines(path: Path, lines: int = 5) -> list[str]:
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8", errors="ignore").splitlines()[-lines:]


def disk_free_gb(path: Path) -> int:
    return int(shutil.disk_usage(path).free // (1024**3))


def query_nvidia_smi(query: str) -> list[str]:
    if shutil.which("nvidia-smi") is None:
        return []
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
    except (subprocess.SubprocessError, FileNotFoundError, PermissionError):
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def command_output(command: list[str]) -> list[str]:
    if not command or shutil.which(command[0]) is None:
        return []
    try:
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
    except (subprocess.SubprocessError, FileNotFoundError, PermissionError):
        return []
    return [line.rstrip() for line in result.stdout.splitlines()]


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def process_info(pid: int) -> str:
    lines = command_output(["ps", "-p", str(pid), "-o", "pid,pcpu,pmem,etime,args", "--no-headers"])
    return lines[0] if lines else ""


def read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def read_output_dir(config_path: Path) -> Path:
    data = read_yaml(config_path)
    raw_output_dir = data.get("output_dir")
    if not raw_output_dir:
        raise SystemExit(f"Config is missing output_dir: {repo_relative(config_path)}")
    output_dir = Path(str(raw_output_dir))
    if output_dir.is_absolute():
        return output_dir
    return REPO_ROOT / output_dir


def remove_nonbest_checkpoints(runs_dir: Path) -> None:
    if not runs_dir.exists():
        return
    for checkpoint_path in runs_dir.rglob("*.pt"):
        if checkpoint_path.name == "best.pt":
            continue
        try:
            checkpoint_path.unlink()
        except OSError:
            continue


def python_candidate_works(candidate: str) -> bool:
    try:
        subprocess.run(
            [candidate, "-c", "import sys; print(sys.executable)"],
            cwd=REPO_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=10,
        )
    except (FileNotFoundError, PermissionError, subprocess.SubprocessError):
        return False
    return True


def detect_python_executable(candidates: list[str] | None = None) -> str:
    ordered_candidates: list[str] = []
    if candidates:
        ordered_candidates.extend(str(candidate) for candidate in candidates)
    ordered_candidates.extend(str(candidate) for candidate in DEFAULT_PYTHON_CANDIDATES)

    seen: set[str] = set()
    for candidate in ordered_candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if python_candidate_works(candidate):
            return candidate

    raise SystemExit(
        "Could not find a usable Python executable. Tried: "
        + ", ".join(candidate for candidate in ordered_candidates if candidate)
    )
