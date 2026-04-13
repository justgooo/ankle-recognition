#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 0 ]]; then
    echo "Usage: $(basename "$0") <command> [args...]" >&2
    exit 2
fi

if ! command -v systemd-run >/dev/null 2>&1; then
    echo "systemd-run is required for host execution." >&2
    exit 1
fi

# Bubblewrap replaces /dev, so GPU-aware commands must be launched from the
# host user session. Keep the caller environment, then pin CUDA ordering so
# host GPU index 0 consistently selects the RTX 3090 wrapper target.
cuda_device_order="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
cuda_visible_devices="${CUDA_VISIBLE_DEVICES:-0}"

env_args=()
passthrough_vars=(
    HOME
    LANG
    LANGUAGE
    LC_ALL
    LD_LIBRARY_PATH
    LOGNAME
    PATH
    PYTHONPATH
    SHELL
    TERM
    USER
    XAUTHORITY
    XDG_RUNTIME_DIR
    http_proxy
    https_proxy
    HTTP_PROXY
    HTTPS_PROXY
    no_proxy
    NO_PROXY
)

for var_name in "${passthrough_vars[@]}"; do
    if [[ -v "${var_name}" ]]; then
        env_args+=("--setenv=${var_name}=${!var_name}")
    fi
done

env_args+=(
    "--setenv=CUDA_DEVICE_ORDER=${cuda_device_order}"
    "--setenv=CUDA_VISIBLE_DEVICES=${cuda_visible_devices}"
)

exec systemd-run \
    --user \
    --quiet \
    --pipe \
    --wait \
    --collect \
    --same-dir \
    "${env_args[@]}" \
    "$@"
