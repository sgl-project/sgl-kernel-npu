#!/usr/bin/env bash
# Build only the pinned upstream LIM OPP; never build DeepEP.
set -eo pipefail

if [[ ${1:-} == --help ]]; then
    echo "Usage: build_original.sh FRESH_OUTPUT_ROOT"
    exit 0
fi
[[ $# -eq 1 ]] || { echo "Usage: build_original.sh FRESH_OUTPUT_ROOT" >&2; exit 2; }
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
output=$1
[[ ! -e "$output" ]] || { echo "Preserve existing build directory: $output" >&2; exit 2; }
mkdir -p -- "$(dirname -- "$output")"
mkdir -- "$output"
output=$(cd -- "$output" && pwd -P)
[[ "$output" != *[[:space:]]* ]] || { echo "CANN OPP build path must not contain whitespace" >&2; exit 2; }
exec > >(tee "$output/build.log") 2>&1
finish() {
    result=$?
    trap - EXIT
    printf '%s\n' "$result" > "$output/exit_code"
    exit "$result"
}
trap finish EXIT
ascend_env=${ASCEND_ENV_PATH:-}
if [[ -z "$ascend_env" ]]; then
    if [[ -f "${ASCEND_HOME_PATH:-}/set_env.sh" ]]; then
        ascend_env="$ASCEND_HOME_PATH/set_env.sh"
    elif [[ -f "${ASCEND_HOME_PATH:-}/bin/setenv.bash" ]]; then
        ascend_env="$ASCEND_HOME_PATH/bin/setenv.bash"
    else
        ascend_env=/usr/local/Ascend/ascend-toolkit/set_env.sh
    fi
fi
python_bin="${PYTHON_BIN:-python3}"
python_bin=$("$python_bin" -c 'import sys; print(sys.executable)')
source "$ascend_env"
export PATH="$(dirname -- "$python_bin"):$PATH"
set -u
python3 "$script_dir/prepare.py" --output-root "$output/prepared"
csrc="$output/prepared/source/csrc"
(cd "$csrc" && bash build.sh --pkg --soc=ascend950 \
    --ops=fused_quant_lightning_indexer_manage \
    --vendor_name=sglc8original "-j${SGL_KERNEL_A5_C8_BUILD_JOBS:-4}")
shopt -s nullglob
installers=("$csrc"/build/*.run)
shopt -u nullglob
[[ ${#installers[@]} -eq 1 ]] || { echo "Expected one original OPP installer" >&2; exit 1; }
mkdir -- "$output/opp"
bash "${installers[0]}" -- "--install-path=$output/opp"
python3 "$script_dir/record_install.py" --build-root "$output"
echo "Original C8 OPP built and privately installed: $output"
