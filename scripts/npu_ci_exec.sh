export PYTHONPATH=\"${GITHUB_WORKSPACE//\\/\/}/python:\${PYTHONPATH:-}\"
exec "$@"
bash "$@"
