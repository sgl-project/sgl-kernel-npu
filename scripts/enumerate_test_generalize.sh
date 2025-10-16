#!/bin/bash
set -e

# 切换目录
if [ -n "${GITHUB_WORKSPACE}" ]; then
    cd "${GITHUB_WORKSPACE}/tests/python/deepep" || { echo "目录不存在"; exit 1; }
fi
# cd "./tests/python/deepep" || { echo "目录不存在"; exit 1; }
# 设置参数范围
indexs=(0 1 2 3)
H_LIST=(7168 6144 2048 4096)
GMM1_HIDDEN_LIST=(2048 2048 768 1536)
SCRIPT="test_fused_deep_moe_accuracy.py"

# 创建临时目录
mkdir -p tmp

# 执行测试
for index in "${indexs[@]}";do

    CMD="python $SCRIPT ${H_LIST[$index]} ${GMM1_HIDDEN_LIST[$index]}"
    echo "Running: $CMD"
    if eval $CMD; then
        echo "测试 H=${H_LIST[$index]} GMM1_HIDDEN=${GMM1_HIDDEN_LIST[$index]} 成功"
    else
        echo "测试 H=${H_LIST[$index]} GMM1_HIDDEN=${GMM1_HIDDEN_LIST[$index]} 失败，退出码: $?"
        exit 1
    fi

    echo "-------------------------------------"
done