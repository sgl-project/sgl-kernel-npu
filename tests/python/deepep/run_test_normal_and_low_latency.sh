#!/bin/bash

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$script_dir" || exit

npu_model=$(npu-smi info 2>/dev/null | grep "910")

if [[ "$npu_model" == *"910B"* ]]; then
    echo "NPU is 910B"
    # set your master node ip
    RANK0_IP=""
    IP=$(hostname -I | awk '{print $1}')

    export WORLD_SIZE=2
    export HCCL_BUFFSIZE=3000
    export HCCL_INTRA_PCIE_ENABLE=1
    export HCCL_INTRA_ROCE_ENABLE=0

    export MASTER_ADDR=${RANK0_IP}
    if [ "${IP}" == "${RANK0_IP}" ]; then
    echo "env rank 0"
    export RANK=0
    else
    echo "env rank 1"
    export RANK=1
    fi
    python test_normal_and_low_latency.py --num-processes 8
else
    echo "NPU is 910C"
    python test_normal_and_low_latency.py --num-processes 16
fi
