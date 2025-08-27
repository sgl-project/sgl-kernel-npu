#! /bin/bash
nproc_per_node=16
master_addr=`hostname -I|awk -F " " '{print$1}'`
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
port=17621

enable_moe_shared=false
if [ "$enable_moe_shared" = true ]; then
    export MOE_SHARED_EXPERT_RANK_NUM=1
else
    export MOE_SHARED_EXPERT_RANK_NUM=0
fi

export MASTER_ADDR=$master_addr
export MASTER_PORT=$port
export LOGLEVEL=INFO

cd "$SCRIPT_DIR" || exit
torchrun --nnodes=1 --nproc-per-node=$nproc_per_node --node_rank=0 --master_addr=$master_addr --master_port=$port ./test_low_latency.py
