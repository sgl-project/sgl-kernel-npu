# torch.ops.catcoc


## Function Description

This is the catcoc(based on catlass) version matmul+comm/comm+matmul fused kernel

Refs: [CATLSS](https://gitcode.com/cann/catlass)  [CATCOC](https://open.codehub.huawei.com/OpenBaize/Ascend/CATCoC)

## Using Cases
### compile support
1. clone catlass in 3rdparty/catlass
2. clone [shmem[coldev]](https://gitee.com/ascend/shmem/tree/coldev/) in 3rdparty and rename examples/templates into 3rdparty/catcoc
3. changing BUILD_CATCOC_OPS=ON in build.sh
4. run 'bash build.sh -a kernels'

### use examples
please check test/python/sgl_kernel_npu/test_catcoc_xxx.py for example

### restrict
1. do not support FP32
2. do not support shmem team
3. do not support dequant fuse(or W8A8)
