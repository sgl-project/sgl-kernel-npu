// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MLAPREPROCESS_TILING_H
#define MLAPREPROCESS_TILING_H

#include <cstdint>

struct PpMatmulTilingData {
    uint32_t numBatch{0};
    uint32_t m{0};
    uint32_t k{0};
    uint32_t n{0};
    uint32_t m0{0};
    uint32_t k0{0};
    uint32_t n0{0};
    uint32_t mLoop{0};
    uint32_t kLoop{0};
    uint32_t nLoop{0};
    uint32_t coreLoop{0};
    uint32_t swizzleCount{0};
    uint32_t swizzleDirect{0};
    uint32_t enShuffleK{0};
    uint32_t blockDim{0};
    uint32_t enLoadAllAmat{0};
    uint32_t b0matPingPongBufferLen{0};
};

struct MlaTilingData {
    uint32_t tilingKey{0};
    uint64_t userWorkspaceSize{0};
    uint64_t s1Offset{0};
    uint64_t s2Offset{0};
    uint64_t s3Offset{0};
    uint64_t s4Offset{0};
    uint64_t s5Offset{0};

    uint32_t numCore{0};
    uint32_t n{0};
    uint32_t perTaskNum{0};
    uint32_t resTaskNum{0};

    PpMatmulTilingData mm1;
    PpMatmulTilingData mm2;
    PpMatmulTilingData mm3;
    // rms1
    uint32_t rmsNumCore1{0};
    uint32_t rmsNumCol1{0};
    uint32_t rmsNumRow1{0};
    uint32_t rmsQuantMin1{0};
    // rms2
    uint32_t rmsNumCore2{0};
    uint32_t rmsNumCol2{0};
    uint32_t rmsNumRow2{0};
    uint32_t rmsQuantMin2{0};

    uint32_t hiddenSizeQ{0};
    uint32_t headNumQ{0};
    uint32_t headDim{0};
    uint32_t concatSize{0};
    uint32_t rotaryCoeff{0};
    uint32_t ntokens{0};
    uint32_t realCore{0};
    uint32_t nlCoreRun{0};
    uint32_t lCoreRun{0};
    uint32_t maxNPerLoopForUb{0};
    uint32_t preCoreLoopTime{0};
    uint32_t preCoreLoopNLast{0};
    uint32_t lastCoreLoopTime{0};
    uint32_t lastCoreLoopNLast{0};

    // EinSumQuant
    uint32_t esqFrontCore{0};
    uint32_t esqTailCore{0};
    uint32_t esqFrontCoreBatch{0};
    uint32_t esqTailCoreBatch{0};
    uint32_t esqHeadNum{0};
    uint32_t esqColNum{0};
    uint32_t esqUbHeadLoop{0};
    uint32_t esqHeadPerLoop{0};
    uint32_t esqHeadTail{0};
    uint32_t esqColLoop{0};
    uint32_t esqColTail{0};
};

#endif  // MLAPREPROCESS_TILING_H