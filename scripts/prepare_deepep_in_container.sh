ENV_DIR="/home/runner/.cache/env_sglang.sh"
source \"$ENV_DIR\"
cd ${GITHUB_WORKSPACE}
bash build.sh
pip install ${GITHUB_WORKSPACE}/output/deep_ep*.whl --no-cache-dir
cd "$(pip show deep-ep | awk '/^Location:/ {print $2}')"
ln -s deep_ep/deep_ep_cpp*.so"
