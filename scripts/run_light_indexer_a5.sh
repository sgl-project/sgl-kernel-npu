#!/usr/bin/env bash
# Use the caller's Torch environment without replacing its installed kernel wheel.
set -eo pipefail
python_bin="${PYTHON:-python3}"
args=()
while (($#)); do
    case "$1" in
        --python)
            if (($# < 2)); then echo '--python requires an interpreter' >&2; exit 2; fi
            python_bin="$2"; shift 2 ;;
        *) args+=("$1"); shift ;;
    esac
done

# Respect an initialized SDK. Otherwise try the supplied machine's SDK first.
if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    for env_file in "${CANN_ENV:-/usr/local/Ascend/cann-9.2.0/set_env.sh}" \
                    /usr/local/Ascend/ascend-toolkit/set_env.sh \
                    /usr/local/Ascend/cann/set_env.sh; do
        if [[ -f "$env_file" ]]; then
            source "$env_file"
            break
        fi
    done
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$python_bin" "$script_dir/light_indexer_a5.py" "${args[@]}"
