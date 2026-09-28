#ifndef INCLUDE_ACT_ARCH_RESOURCE_HPP
#define INCLUDE_ACT_ARCH_RESOURCE_HPP

#include "../../act/act.hpp"
#include "../../act/arch/local_tensor_buffer.hpp"

namespace Act::Arch {

template <class ArchTag>
struct Resource {
public:
    AscendC::TPipe pipe;

    LocalTensorBuffer<ArchTag, AscendC::TPosition::A1> l1Buf;
    LocalTensorBuffer<ArchTag, AscendC::TPosition::A2> l0ABuf;
    LocalTensorBuffer<ArchTag, AscendC::TPosition::B2> l0BBuf;
    LocalTensorBuffer<ArchTag, AscendC::TPosition::CO1> l0CBuf;
    LocalTensorBuffer<ArchTag, AscendC::TPosition::VECCALC> ubBuf;

    ACT_DEVICE
    Resource()
    {
        // The initialization of AscendC::Tpipe will insert some synchronization
        // interfaces, which may conflict with the usage by users. Therefore, the
        // "destroy" interface is used for releasing.
        pipe.Destroy();
    }
};

}  // namespace Act::Arch

#endif  // INCLUDE_ACT_ARCH_RESOURCE_HPP
