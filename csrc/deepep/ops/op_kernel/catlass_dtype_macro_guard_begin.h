#ifndef OPS_OP_KERNEL_CATLASS_DTYPE_MACRO_GUARD_BEGIN_H
#define OPS_OP_KERNEL_CATLASS_DTYPE_MACRO_GUARD_BEGIN_H

// Some CANN/AscendC headers export DT_* as global macros. Upstream catlass also
// refers to namespaced AscendC::DT_* identifiers, so these global macros must be
// hidden while parsing the catlass include chain.

namespace deepep_catlass_dtype_saved {
#if defined(DT_FLOAT)
#define DEEPEP_CATLASS_SAVED_DT_FLOAT
#define DEEPEP_CATLASS_DT_FLOAT_VALUE DT_FLOAT
#define DEEPEP_CATLASS_RESTORE_DT_FLOAT
enum : uint32_t { kDtFloat = DEEPEP_CATLASS_DT_FLOAT_VALUE };
#endif

#if defined(DT_BF16)
#define DEEPEP_CATLASS_SAVED_DT_BF16
#define DEEPEP_CATLASS_DT_BF16_VALUE DT_BF16
#define DEEPEP_CATLASS_RESTORE_DT_BF16
enum : uint32_t { kDtBf16 = DEEPEP_CATLASS_DT_BF16_VALUE };
#endif

#if defined(DT_FLOAT16)
#define DEEPEP_CATLASS_SAVED_DT_FLOAT16
#define DEEPEP_CATLASS_DT_FLOAT16_VALUE DT_FLOAT16
#define DEEPEP_CATLASS_RESTORE_DT_FLOAT16
enum : uint32_t { kDtFloat16 = DEEPEP_CATLASS_DT_FLOAT16_VALUE };
#endif
}  // namespace deepep_catlass_dtype_saved

#undef DT_FLOAT
#undef DT_BF16
#undef DT_FLOAT16

#undef DEEPEP_CATLASS_DT_FLOAT_VALUE
#undef DEEPEP_CATLASS_DT_BF16_VALUE
#undef DEEPEP_CATLASS_DT_FLOAT16_VALUE

#if defined(DEEPEP_CATLASS_A5_COMPAT)
namespace AscendC {
#ifdef DEEPEP_CATLASS_SAVED_DT_FLOAT
static constexpr uint32_t DT_FLOAT = deepep_catlass_dtype_saved::kDtFloat;
#endif
#ifdef DEEPEP_CATLASS_SAVED_DT_BF16
static constexpr uint32_t DT_BF16 = deepep_catlass_dtype_saved::kDtBf16;
#endif
#ifdef DEEPEP_CATLASS_SAVED_DT_FLOAT16
static constexpr uint32_t DT_FLOAT16 = deepep_catlass_dtype_saved::kDtFloat16;
#endif
}  // namespace AscendC
#endif

#endif
