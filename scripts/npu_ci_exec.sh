ENV_DIR="/usr/local/Ascend/ascend-toolkit/set_env.sh"
echo "GITHUB_WORKSPACE=${GITHUB_WORKSPACE}"

source \"$ENV_DIR\"
export PYTHONPATH=\"${GITHUB_WORKSPACE//\\/\/}/python:\${PYTHONPATH:-}\"
exec "$@"
bash "$@"
