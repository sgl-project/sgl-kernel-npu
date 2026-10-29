#include "cache_loc_tiling.h"
#include "sgl_operator_config.h"

namespace {
ge::graphStatus Tiling(gert::TilingContext *context)
{
    return sgl_cache_loc::BuildTiling(context, false);
}
}  // namespace

namespace ops {
class SglCacheLocAssign : public OpDef
{
public:
    explicit SglCacheLocAssign(const char *name) : OpDef(name)
    {
        this->Input("req_indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("token_pool")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("start_offset")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("end_offset")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("out_cache_loc")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        // A same-name output generates a Ref argument instead of a new allocation.
        this->Output("token_pool")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->SetInferShape(sgl_cache_loc::InferShape<1>).SetInferDataType(sgl_cache_loc::InferDataType<1>);
        this->AICore().SetTiling(Tiling);
        for (const auto *config : SGL_CANN_CONFIGS) {
            this->AICore().AddConfig(config);
        }
    }
};
OP_ADD(SglCacheLocAssign);
}  // namespace ops
