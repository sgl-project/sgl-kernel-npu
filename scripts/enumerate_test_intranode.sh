#!/bin/bash

# 默认值
SKIP_BUILD=false

TEMP=$(getopt -o sw:t:h --long skip-build -n "$0" -- "$@")
if [ $? != 0 ]; then
    echo "Terminating..." >&2
    exit 1
fi

eval set -- "$TEMP"

while true; do
    case "$1" in
        -s|--skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Invalid option: $1" >&2
            show_help
            exit 1
            ;;
    esac
done

# 切换目录
cd ${GITHUB_WORKSPACE}

# 条件构建
if [ "$SKIP_BUILD" = false ]; then
    echo ">>> Building package..."
    bash build.sh -a deepep || { echo "Build failed!"; exit 1; }
    pip uninstall -y deep-ep
    pip install ./output/deep_ep-*.whl || { echo "Install failed!"; exit 1; }
else
    echo ">>> Skipping build and install (--skip-build)"
fi

# 进入测试目录
cd ./tests/python/deepep || { echo "Test directory not found"; exit 1; }

# 设置 Ascend 环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh

#遍历test_intranode.py
# 设置参数范围
NUM_PROCESSES_LIST_=(4 8 16)
NUM_TOKENS_LIST=(1024 2048 4096)
HIDDEN_LIST=(4096 7168)
NUM_TOPK_LIST=(4 8)
NUM_EXPERTS_LIST=(64 128 256)
ACTIVE_RANKS_LIST=("" "0,1" "0,2,3")
ENABLE_DIAGNOSE_LIST=("false" "true")

SCRIPT="test_intranode.py"

# 遍历所有组合
for NUM_PROCESSES in "${NUM_PROCESSES_LIST[@]}"; do
  for NUM_TOKENS in "${NUM_TOKENS_LIST[@]}"; do
    for HIDDEN in "${HIDDEN_LIST[@]}"; do
      for NUM_TOPK in "${NUM_TOPK_LIST[@]}"; do
        for NUM_EXPERTS in "${NUM_EXPERTS_LIST[@]}"; do
          for ACTIVE_RANKS in "${ACTIVE_RANKS_LIST[@]}"; do
            for ENABLE_DIAGNOSE in "${ENABLE_DIAGNOSE_LIST[@]}"; do

              # 构建命令
              CMD="python3 $SCRIPT \
                --num-processes $NUM_PROCESSES \
                --num-tokens $NUM_TOKENS \
                --hidden $HIDDEN \
                --num-topk $NUM_TOPK \
                --num-experts $NUM_EXPERTS"

              # 添加可选参数
              if [ -n "$ACTIVE_RANKS" ]; then
                CMD="$CMD --active-ranks \"$ACTIVE_RANKS\""
              fi

              if [ "$ENABLE_DIAGNOSE" == "true" ]; then
                CMD="$CMD --enable-diagnose"
              fi

              # 打印并执行命令
              echo "Running: $CMD"
              eval $CMD

              echo "--------------------------------------------------"

            done
          done
        done
      done
    done
  done
done
