ENV_DIR="/home/runner/.cache/env_sglang.sh"
echo "GITHUB_WORKSPACE=${GITHUB_WORKSPACE}"
docker exec \
  sglang_ci_a3 \
  /bin/bash -c "source \"$ENV_DIR\" && export PYTHONPATH=\"${GITHUB_WORKSPACE//\\/\/}/python:\${PYTHONPATH:-}\" && exec \"\$@\"" \
  bash "$@"