# set your master node ip
IP=$(hostname -I | awk '{print $1}')

export HCCL_BUFFSIZE=3000
export HCCL_INTRA_PCIE_ENABLE=1
export HCCL_INTRA_ROCE_ENABLE=0

export DEEPEP_SHMEM_ENABLE=0  # Option: 0: without shmem ; 1: with shmem

# export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15

python test_low_latency.py
