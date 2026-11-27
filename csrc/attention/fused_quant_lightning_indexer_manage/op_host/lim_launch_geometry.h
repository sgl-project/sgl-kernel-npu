// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <limits>
#include "../op_kernel/arch35/fused_quant_lightning_indexer_manage_tiling.h"
#include "../op_kernel/arch35/fused_quant_lightning_indexer_manage_workspace.h"
#include "../op_kernel/arch35/lim_payload_layout.h"

inline bool InitQuantLimGeometry(FusedQuantLightningIndexerManageTilingData &t)
{
    const uint64_t routes = std::max<uint64_t>(4ULL * t.batchSize, t.totalQueryRows);
    const auto segments = fused_quant_li_manage_workspace::CandidateSegmentCapacity(t.sourceCapacity, t.usedCoreNum);
    const auto bytes = fused_quant_li_manage_workspace::TotalBytes(
        t.sourceCapacity, routes, segments, t.sourceCapacity > lim_payload_layout::PACKED_SOURCE_CAPACITY);
    if (routes > UINT32_MAX || bytes > UINT32_MAX) return false;
    t.managementWorkspaceOffset = bytes;
    t.scoreWorkspaceStride = 4 * t.sourceCapacity * sizeof(uint16_t);
    t.scoreRowStride = t.sourceCapacity;
    t.queryTileSize = t.indexHeads == 32 && (t.totalQueryRows > 64 || t.batchSize * 2ULL > t.usedCoreNum) ? 4 : 2;
    t.routeStorageRows = routes;
    return true;
}

inline uint64_t QuantLimWorkspaceBytes(const FusedQuantLightningIndexerManageTilingData &t, uint64_t systemBytes)
{
    return systemBytes + t.managementWorkspaceOffset + (uint64_t(t.batchSize) + t.totalQueryRows) * 32;
}
