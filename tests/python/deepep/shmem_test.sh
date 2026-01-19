export SHMEM_UID_SESSION_ID=127.0.0.1:12345
# export NPU_SHMEM_SYMMETRIC_SIZE='8G'
export DEEPEP_SHMEM_ENABLE=1
export SHMEM_SYMMETRIC_SIZE=8000

export HCCL_BUFFSIZE=200
rm -rf ./logs
export ASCEND_PROCESS_LOG_PATH=./logs
export ASCEND_GLOBAL_LOG_LEVEL=3
# export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# export SHMEM_LOG_LEVEL=DEBUG
# export SHMEM_LOG_TO_STDOUT=1

# export DEEP_NORMAL_MODE_USE_INT8_QUANT=1

# python test_shmem_intranode.py --num-processes=8 --num-tokens=8 --num-topk=8 --num-experts=32

# python test_shmem_intranode.py --num-processes=8 --num-tokens=1024 --num-topk=8 --num-experts=256

# python test_shmem_intranode.py --num-processes=8 --num-tokens=2048 --num-topk=8 --num-experts=256

# python test_shmem_intranode.py --num-processes=8 --num-tokens=4096 --num-topk=8 --num-experts=256

# python test_shmem_intranode.py --num-processes=8 --num-tokens=8000 --num-topk=8 --num-experts=256

# python test_shmem_intranode.py --num-processes=8 --num-tokens=16000 --num-topk=8 --num-experts=256


# test_shmem_intranode_api.py
# python test_shmem_intranode_api.py --num-processes=8 --num-tokens=8 --num-topk=8 --num-experts=32

python test_shmem_intranode_api.py --num-processes=8 --num-tokens=1024 --num-topk=8 --num-experts=256



