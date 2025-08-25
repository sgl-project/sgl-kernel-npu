export HCCL_BUFFSIZE=2048
export PYTHONPATH=\"${GITHUB_WORKSPACE//\\/\/}/python:\${PYTHONPATH:-}\"
exec "$@"
bash "$@"
