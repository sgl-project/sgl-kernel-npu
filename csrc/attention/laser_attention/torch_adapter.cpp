// The LaserAttention implementation and its ACLNN dispatch are the legacy
// attentions plugin, copied verbatim under legacy_plugin/. They are compiled
// here as one translation unit so the build declaration stays a single source.
//
// The only adaptation is the registration namespace: the legacy extension
// exported `attentions::la`, while this repository exposes operators through
// `torch.ops.npu.*`. The schema string is copied from the legacy
// register_ops.cpp for the `la` entry, so Python sees no interface change.
#include "legacy_plugin/find_op_path.cpp"
#include "legacy_plugin/la.cpp"

#include <torch/library.h>

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def(
        "la(Tensor query, Tensor key, Tensor value, Tensor? atten_mask=None, Tensor? alibi_mask=None, "
        "Tensor? drop_mask=None, float scale_value=1.0, int head_num=2, str input_layout='BNSD', "
        "float keep_prob=1.0, int pre_tokens=2147483647, int next_tokens=1, bool is_highPrecision=True) "
        "-> (Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("la", &la);
}
