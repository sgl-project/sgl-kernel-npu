#!/bin/bash
CURRENT_DIR=$(
    cd $(dirname ${BASH_SOURCE:-$0})
    pwd
)
TEST_ROOT=$(dirname ${CURRENT_DIR})


RANK_SIZE="2"
IPPORT="tcp://127.0.0.1:8776"
GNPU_NUM="2"
FIRST_NPU="0"
FIRST_RANK="0"
TEST_TYPE="float"
ZERO_BUFF=0

ZCCL_LIB_DIR=${TEST_ROOT}/zccl/lib

# Golden generate
rm -rf golden output
mkdir -p golden output
python3 ./scripts/data_gen.py $RANK_SIZE $TEST_TYPE

# Kernel test
export LD_LIBRARY_PATH=${ZCCL_LIB_DIR}:${TEST_ROOT}/out/lib:${SHMEM_HOME_PATH}/shmem/lib/:${SHMEM_HOME_PATH}/memfabric_hybrid/lib/:${ASCEND_HOME_PATH}/lib64:$LD_LIBRARY_PATH
pids=()
for (( idx =0; idx < ${GNPU_NUM}; idx = idx + 1 )); do
    msprof --application="${TEST_ROOT}/out/bin/ascendc_all_gather $RANK_SIZE $idx $IPPORT $GNPU_NUM $FIRST_RANK $FIRST_NPU $TEST_TYPE $ZERO_BUFF" --output=${CURRENT_DIR}/output/ &
    pid=$!
    pids+=("$pid")
    echo "$pid background process recorded"
done

ret=0
for pid in ${pids[@]}; do
    wait $pid
    echo "wait process $pid done"
    cur_ret=$?
    if [[ $cur_ret -ne 0 ]]; then
        ret=$cur_ret
    fi
done
