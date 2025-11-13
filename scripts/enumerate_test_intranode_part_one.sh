#!/bin/bash

# 切换目录
cd ${GITHUB_WORKSPACE}/tests/python/deepep

#遍历test_intranode.py
# 设置参数范围
NUM_PROCESSES_LIST=(8)
NUM_TOKENS_LIST=(1 4096)
HIDDEN_LIST=(4096)
NUM_TOPK_LIST=(8)
NUM_EXPERTS_LIST=(64)
ACTIVE_RANKS_LIST=("" "0,1")
ENABLE_DIAGNOSE_LIST=("false")

SCRIPT="test_intranode.py"

for NUM_PROCESSES in "${NUM_PROCESSES_LIST[@]}"; do
  for NUM_TOKENS in "${NUM_TOKENS_LIST[@]}"; do
    for HIDDEN in "${HIDDEN_LIST[@]}"; do
      for NUM_TOPK in "${NUM_TOPK_LIST[@]}"; do
        for NUM_EXPERTS in "${NUM_EXPERTS_LIST[@]}"; do
          for ACTIVE_RANKS in "${ACTIVE_RANKS_LIST[@]}"; do
            for ENABLE_DIAGNOSE in "${ENABLE_DIAGNOSE_LIST[@]}"; do
              CMD="python3 $SCRIPT \
                --num-processes $NUM_PROCESSES \
                --num-tokens $NUM_TOKENS \
                --hidden $HIDDEN \
                --num-topk $NUM_TOPK \
                --num-experts $NUM_EXPERTS"

              if [ -n "$ACTIVE_RANKS" ]; then
                CMD="$CMD --active-ranks \"$ACTIVE_RANKS\""
              fi

              if [ "$ENABLE_DIAGNOSE" == "true" ]; then
                CMD="$CMD --enable-diagnose"
              fi

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

cd ./
