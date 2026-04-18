/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once
#include "kernel_operator.h"

using namespace AscendC;
#define PIPE_FIX (pipe_t)10

#define ALLCUBE_READY(iiii, append_to_pipe) CrossCoreSetFlag<0x0, append_to_pipe>(iiii)
#define ALLCUBE_WAIT(iiii) CrossCoreWaitFlag(iiii)
#define ALLVEC_READY(iiii, append_to_pipe) CrossCoreSetFlag<0x0, append_to_pipe>(iiii)
#define ALLVEC_WAIT(iiii) CrossCoreWaitFlag(iiii)
#define CUBE_READY(iiii, append_to_pipe) CrossCoreSetFlag<0x2, append_to_pipe>(iiii)
#define WAIT_CUBE(iiii) CrossCoreWaitFlag(iiii)
#define VEC_READY(iiii, append_to_pipe) CrossCoreSetFlag<0x2, append_to_pipe>(iiii)
#define WAIT_VEC(iiii) CrossCoreWaitFlag(iiii)

constexpr uint64_t VECTORFULLMASK[2] = {(uint64_t)-1, (uint64_t)-1};
constexpr int SGL_TWO = 2;  // vector per core; dbuff
constexpr int SGL_THREE = 3;
constexpr int SGL_FOUR = 4;

constexpr int M_BLK_SIZE = 16;
constexpr int N_BLK_SIZE = 32;
constexpr int SGL_BLK_SIZE = 64;

constexpr int MTE_FLOAT = 8;
constexpr int MTE_HALF = 16;
constexpr int VEC_HALF = 128;

constexpr int NUM_ELE_PERBLK_FLOAT = 8;

__aicore__ constexpr HardEvent GetHardEventByPipe(pipe_t src, pipe_t dst)
{
    if (src == PIPE_MTE2) {
        if (dst == PIPE_MTE1) {
            return HardEvent::MTE2_MTE1;
        } else if (dst == PIPE_V) {
            return HardEvent::MTE2_V;
        }
    } else if (src == PIPE_MTE1) {
        if (dst == PIPE_MTE2) {
            return HardEvent::MTE1_MTE2;
        } else if (dst == PIPE_M) {
            return HardEvent::MTE1_M;
        } else if (dst == PIPE_FIX) {
            return HardEvent::MTE1_FIX;
        }
    } else if (src == PIPE_M) {
        if (dst == PIPE_MTE1) {
            return HardEvent::M_MTE1;
        } else if (dst == PIPE_FIX) {
            return HardEvent::M_FIX;
        }
    } else if (src == PIPE_FIX) {
        if (dst == PIPE_M) {
            return HardEvent::FIX_M;
        } else if (dst == PIPE_MTE1) {
            return HardEvent::FIX_MTE1;
        }
    } else if (src == PIPE_V) {
        if (dst == PIPE_MTE2) {
            return HardEvent::V_MTE2;
        } else if (dst == PIPE_MTE3) {
            return HardEvent::V_MTE3;
        }
    } else if (src == PIPE_MTE3) {
        if (dst == PIPE_V) {
            return HardEvent::MTE3_V;
        }
    }
}

__aicore__ inline void OccupyMMTE1Events()
{
    if ASCEND_IS_AIC {
        TPipe *pipe_ptr = GetTPipePtr();
        pipe_ptr->AllocEventID<HardEvent::M_MTE1>();
        pipe_ptr->AllocEventID<HardEvent::M_MTE1>();
        pipe_ptr->AllocEventID<HardEvent::M_MTE1>();
    }
}

__aicore__ constexpr int Align16B(int x)
{
    return (x + (M_BLK_SIZE - 1)) / M_BLK_SIZE * M_BLK_SIZE;
}

__aicore__ constexpr int Align32B(int x)
{
    return (x + (N_BLK_SIZE - 1)) / N_BLK_SIZE * N_BLK_SIZE;
}

__aicore__ constexpr int Align64B(int x)
{
    return (x + (SGL_BLK_SIZE - 1)) / SGL_BLK_SIZE * SGL_BLK_SIZE;
}

__aicore__ inline int CeilDiv(int a, int b)
{
    return (a + b - 1) / b;
}

template <typename T, typename T1, typename T2>
__aicore__ inline T1 shiftAddr(T1 base, uint64_t size, T2 &offset)
{
    auto res = base + offset;
    offset += size * sizeof(T);
    return res;
}

/* ------------- Tensor ------------- */
template <TPosition pos, typename T>
__aicore__ inline void AllocateLocalTensor(LocalTensor<T> &tsr, int len)
{
    TBuf<pos> tbuf;
    TPipe *ptr = GetTPipePtr();
    ptr->InitBuffer(tbuf, len * sizeof(T));
    tsr = tbuf.template Get<T>();
}

/* ------------- Double Buffer ------------- */
template <typename T, TPosition pos>
class DBuff
{
public:
    __aicore__ inline DBuff() {}
    __aicore__ inline void Init(int len)
    {
        TPipe *ptr = GetTPipePtr();
        ptr->InitBuffer(buf1, len * sizeof(T));
        ptr->InitBuffer(buf2, len * sizeof(T));
        tsr1 = buf1.template Get<T>();
        tsr2 = buf2.template Get<T>();
    }

    __aicore__ inline LocalTensor<T> get(int i)
    {
        if (i % SGL_TWO == 0) {
            return tsr1;
        } else {
            return tsr2;
        }
    }

private:
    TBuf<pos> buf1, buf2;
    LocalTensor<T> tsr1, tsr2;
};

/* ------------- Triple Buffer ------------- */
template <typename T, TPosition pos>
class TBuff
{
public:
    __aicore__ inline TBuff() {}
    __aicore__ inline void Init(int len)
    {
        TPipe *ptr = GetTPipePtr();
        ptr->InitBuffer(buf1, len * sizeof(T));
        ptr->InitBuffer(buf2, len * sizeof(T));
        ptr->InitBuffer(buf3, len * sizeof(T));
        tsr1 = buf1.template Get<T>();
        tsr2 = buf2.template Get<T>();
        tsr3 = buf3.template Get<T>();
    }

    __aicore__ inline LocalTensor<T> get(int i)
    {
        if (i % SGL_THREE == 0) {
            return tsr1;
        } else if (i % SGL_THREE == 1) {
            return tsr2;
        } else {
            return tsr3;
        }
    }

private:
    TBuf<pos> buf1, buf2, buf3;
    LocalTensor<T> tsr1, tsr2, tsr3;
};

/* ------------- Events ------------- */
template <HardEvent hardevent_t>
class SEvent
{
public:
    __aicore__ inline SEvent() {}
    __aicore__ inline void Init()
    {
        TPipe *pipe_ptr = GetTPipePtr();
        id1 = (event_t)pipe_ptr->AllocEventID<hardevent_t>();
    }
    __aicore__ inline void wait()
    {
        AscendC::WaitFlag<hardevent_t>(id1);
    }
    __aicore__ inline void set()
    {
        AscendC::SetFlag<hardevent_t>(id1);
    }
    __aicore__ inline void setall()
    {
        set();
    }
    __aicore__ inline void release()
    {
        wait();
    }

private:
    event_t id1;
};

template <pipe_t p1, pipe_t p2>
class DEvent
{
public:
    __aicore__ inline DEvent() {}
    __aicore__ inline void Init()
    {
        TPipe *pipe_ptr = GetTPipePtr();
        id1 = (event_t)pipe_ptr->AllocEventID<GetHardEventByPipe(p1, p2)>();
        id2 = (event_t)pipe_ptr->AllocEventID<GetHardEventByPipe(p1, p2)>();
    }
    __aicore__ inline void wait()
    {
        if (wait_cnt % SGL_TWO == 0) {
            wait_flag(p1, p2, id1);
        } else {
            wait_flag(p1, p2, id2);
        }
        wait_cnt++;
    }
    __aicore__ inline void set()
    {
        if (set_cnt % SGL_TWO == 0) {
            set_flag(p1, p2, id1);
        } else {
            set_flag(p1, p2, id2);
        }
        set_cnt++;
    }
    __aicore__ inline void setall()
    {
        set();
        set();
    }
    __aicore__ inline void release()
    {
        for (int i = wait_cnt; i < set_cnt; ++i) {
            wait();
        }
    }

private:
    event_t id1, id2;
    int wait_cnt = 0;
    int set_cnt = 0;
};
