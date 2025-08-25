ENV_DIR="/usr/local/Ascend/ascend-toolkit/set_env.sh"

source \"$ENV_DIR\"

export PYTHONPATH=\"${GITHUB_WORKSPACE//\\/\/}/python:\${PYTHONPATH:-}\"
exec "$@"
bash "$@"
