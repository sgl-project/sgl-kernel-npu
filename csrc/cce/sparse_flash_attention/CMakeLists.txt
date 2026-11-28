# Raw CCE operator for Ascend 950 (arch 3510): a bisheng-compiled sparse flash
# attention kernel. The HOST interface in op_host/ selects one of four bisheng
# builds of the same op_kernel/sfa.cce source (one per head count); the compiled
# objects are archived into libsgl_cce_operators.a and linked into
# libsgl_kernel_npu.so.
#
# The kernel symbol and its launcher are macro-renamed per variant so the four
# objects coexist in one archive (mirrors the reference build.sh).
sgl_register_operator(
    ID cce_sparse_flash_attention
    EXECUTION_UNIT CCE
    SUPPORTED_ARCHS 3510
    CPP_SOURCES op_host/sfa_dispatch.cpp
    TORCH_ADAPTER torch_adapter.cpp
)
foreach(heads IN ITEMS 16 32 64 128)
    if(heads LESS 64)
        set(bm "${heads}")
        set(bn 64)
        set(extra_definitions)
    else()
        set(bm 64)
        set(bn 128)
        set(extra_definitions CCE_MTE2_SCATTER)
        if(heads EQUAL 128)
            list(APPEND extra_definitions CCE_WIDE_QK)
        endif()
    endif()
    sgl_register_cce_kernel(
        ID cce_sparse_flash_attention
        ARCH 3510
        VARIANT h${heads}
        SOURCE op_kernel/sfa.cce
        COMPILE_DEFINITIONS
            CCE_BM=${bm}
            CCE_BN=${bn}
            ${extra_definitions}
            sfa=sfa_h${heads}
            cce_launch_sfa=cce_launch_sfa_h${heads}
    )
endforeach()
