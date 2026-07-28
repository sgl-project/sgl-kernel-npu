/**
 * @file unidex_copy.cpp
 * @brief 基于稀疏索引的通用行拷贝算子（Ascend AI Core）
 *
 * 算子语义：
 * -------------------
 * 根据稀疏索引映射关系，从源缓冲区拷贝行数据到目标缓冲区。
 * 对于每个有效的映射条目 i（valid_mask[i] == true）：
 *   dst[dst_index[i]] = src[src_index[i]]
 *
 * 典型使用场景：
 * ----------------
 * - 稀疏 KV 缓存检索：src 包含 128k*n 个缓存的 key-value 对
 * - 索引形状：[batch_size, topk]，其中 batch_size ≈8，topk ≈2048
 * - 源索引稀疏/随机（gather 模式）
 * - 目标索引通常连续（scatter 模式）
 * - 命中率由 valid_mask 控制（通常 0.0-1.0）
 *
 * 内存布局：
 * -------------
 * - src 和 dst 均视为行主序字节缓冲
 * - 每个逻辑行占用恰好 blockBytes 字节
 * - srcRows：src 缓冲区总行数
 * - dstRows：dst 缓冲区总行数
 * - 行访问：src[row_idx] = src_base + row_idx * blockBytes
 *
 * 性能特征：
 * ---------------------------
 * - 任务分配：maxCopy 个条目在 AI Core 间均分
 * - 每个核心处理 copyRowsPerCore = ceil(maxCopy / 核心数) 个映射
 * - 稀疏 valid_mask 可能导致负载不均（某些核心跳过大量条目）
 * - 行拷贝：单阶段通过 UB 使用双缓冲乒乓完成
 * - DMA 粒度：字节级（uint8_t）以支持最大的数据类型灵活性
 *
 * 约束与限制：
 * -------------------------
 * - blockBytes 必须适配单个 UB 缓冲分配（当前：整行一次传输）
 * - 对于非常大的 blockBytes（如 >32KB），应考虑分块行拷贝
 * - blockBytes 理想情况下应为 32 字节对齐以获得最佳 DMA 性能
 * - 无自动索引压缩：无效条目仍消耗核心迭代周期
 * - 假设 src 和 dst 具有相同的 blockBytes（由 host 包装层强制）
 *
 * 缓冲管理：
 * -----------------
 * - BUFFER_NUM=2：启用 copy-in 和 copy-out 操作间的乒乓
 * - 当一个缓冲拷出到 GM 时，另一个可以从 GM 拷入
 * - 通过重叠 GM<->UB 传输提升吞吐量
 * - 最小队列深度（2）平衡吞吐量与 UB 内存消耗
 *
 * @note 此算子替代 index_copy_dtd.cpp，移除了硬编码的 2 字节元素假设
 * @note 通过字节级拷贝支持所有数据类型（blockBytes 参数决定行大小）
 */

#include "kernel_operator.h"

using CopyUnit = uint8_t;
constexpr uint32_t BUFFER_NUM = 2;

class KernelUniDexCopy
{
public:
    __aicore__ inline KernelUniDexCopy() {}

    /**
     * @brief 使用缓冲指针和拷贝参数初始化算子
     *
     * @param src 源缓冲区基地址（GM） b,s,n,d/ t,n,d   n:head d:dim
     * @param dst 目标缓冲区基地址（GM） b,s,n,d/ t,n,d   n:head d:dim    addr 指示哪两个维度合并
     * @param src_index 源行索引数组（GM，int64_t，长度=maxCopy）
     * @param dst_index 目标行索引数组（GM，int64_t，长度=maxCopy）
     * @param valid_mask 有效性掩码数组（GM，uint8_t/bool，长度=maxCopy）
     * @param srcRows 源缓冲区总行数
     * @param dstRows 目标缓冲区总行数
     * @param blockBytes 每个逻辑行的字节数（src 和 dst 必须相同）
     * @param maxCopy 要处理的映射条目总数
     * @param pipeIn 用于队列管理的 TPipe 指针
     *
     * @note 在 Process() 之前每个 AI Core 块调用一次
     * @note 任务分配：本核心处理 [blockIdx * copyRowsPerCore, min(maxCopy, (blockIdx+1) * copyRowsPerCore))
     */
    __aicore__ inline void Init(GM_ADDR src, GM_ADDR dst, GM_ADDR src_index, GM_ADDR dst_index, GM_ADDR valid_mask,
                                uint32_t srcRows, uint32_t dstRows, uint32_t blockBytes, uint32_t maxCopy,
                                AscendC::TPipe *pipeIn)
    {
        this->srcRows = srcRows;
        this->dstRows = dstRows;
        this->blockBytes = blockBytes;
        this->maxCopy = maxCopy;
        this->pipe = pipeIn;

        const uint32_t blockNum = AscendC::GetBlockNum();
        this->copyRowsPerCore = (maxCopy + blockNum - 1) / blockNum;

        srcGm.SetGlobalBuffer((__gm__ CopyUnit *)src, srcRows * blockBytes);
        dstGm.SetGlobalBuffer((__gm__ CopyUnit *)dst, dstRows * blockBytes);
        srcIndexGm.SetGlobalBuffer((__gm__ int64_t *)src_index, maxCopy);
        dstIndexGm.SetGlobalBuffer((__gm__ int64_t *)dst_index, maxCopy);
        validMaskGm.SetGlobalBuffer((__gm__ uint8_t *)valid_mask, maxCopy);

        this->alignedBlockBytes = (blockBytes + 31U) & ~31U;
        pipe->InitBuffer(copyQue, BUFFER_NUM, alignedBlockBytes * sizeof(CopyUnit));
    }

    /**
     * @brief 本 AI Core 的主处理循环
     *
     * 处理策略：
     * 1. 确定本核心的映射范围：[coreBegin, coreEnd)
     * 2. 对于范围内的每个映射索引 i：
     *    a. 检查 valid_mask[i]，若无效则跳过
     *    b. 验证 src_index[i] 和 dst_index[i] 在边界内
     *    c. 当缓冲可用时，入队从 src[src_index[i]] 的 copy-in
     *    d. 当缓冲满时，出队并 copy-out 到 dst[dst_index[i]]
     * 3. 刷新剩余缓冲的行
     *
     * 乒乓行为：
     * - 队列最多在 UB 中容纳 BUFFER_NUM 行
     * - 队列满时：必须先 copy-out 才能接受新的 copy-in
     * - 这在迭代间重叠了 GM 读取（copy-in）和 GM 写入（copy-out）
     *
     * 负载均衡问题：
     * - 稀疏 valid_mask 导致每个核心的实际工作不均
     * - 具有大量无效条目的核心提前完成但仍需迭代
     * - 替代方案：在 host 端压缩有效索引（权衡：额外的 host 开销）
     */
    __aicore__ inline void Process()
    {
        if (blockBytes == 0) {
            return;
        }

        const uint32_t coreBegin = AscendC::GetBlockIdx() * copyRowsPerCore;
        uint32_t coreEnd = coreBegin + copyRowsPerCore;
        if (coreEnd > maxCopy) {
            coreEnd = maxCopy;
        }

        uint32_t dstOffsets[BUFFER_NUM] = {0, 0};
        uint32_t queueHead = 0;
        uint32_t queueTail = 0;
        uint32_t queued = 0;

        for (uint32_t i = coreBegin; i < coreEnd; ++i) {
            uint32_t srcOffset = 0;
            uint32_t dstOffset = 0;
            if (!BuildCopyTask(i, srcOffset, dstOffset)) {
                continue;
            }

            if (queued == BUFFER_NUM) {
                CopyOut(dstOffsets[queueHead]);
                queueHead = NextQueueIndex(queueHead);
                --queued;
            }

            CopyIn(srcOffset);
            dstOffsets[queueTail] = dstOffset;
            queueTail = NextQueueIndex(queueTail);
            ++queued;
        }

        while (queued > 0) {
            CopyOut(dstOffsets[queueHead]);
            queueHead = NextQueueIndex(queueHead);
            --queued;
        }
    }

private:
    /**
     * @brief 循环队列索引递增
     * @param index 当前队列索引
     * @return 下一个队列索引（在 BUFFER_NUM-1 后回绕到 0）
     */
    __aicore__ inline uint32_t NextQueueIndex(uint32_t index) const
    {
        return index == BUFFER_NUM - 1 ? 0 : index + 1;
    }

    /**
     * @brief 验证并构建映射条目的拷贝任务
     *
     * @param mapIdx src_index/dst_index/valid_mask 数组中的索引
     * @param[out] srcOffset src 缓冲区中的字节偏移（row_idx * blockBytes）
     * @param[out] dstOffset dst 缓冲区中的字节偏移（row_idx * blockBytes）
     * @return 如果应继续拷贝返回 true，如果无效/越界返回 false
     *
     * 验证规则：
     * 1. valid_mask[mapIdx] 必须非零
     * 2. src_index[mapIdx] 必须在 [0, srcRows) 范围内
     * 3. dst_index[mapIdx] 必须在 [0, dstRows) 范围内
     * 4. 拒绝负索引（为安全显式检查）
     *
     * @note 越界索引被静默跳过（无错误报告）
     * @note host 包装层应验证索引以提供更好的错误消息
     */
    __aicore__ inline bool BuildCopyTask(uint32_t mapIdx, uint32_t &srcOffset, uint32_t &dstOffset)
    {
        if (validMaskGm.GetValue(mapIdx) == 0) {
            return false;
        }

        const int64_t srcRow = srcIndexGm.GetValue(mapIdx);
        const int64_t dstRow = dstIndexGm.GetValue(mapIdx);
        if (srcRow < 0 || dstRow < 0) {
            return false;
        }
        if (srcRow >= static_cast<int64_t>(srcRows) || dstRow >= static_cast<int64_t>(dstRows)) {
            return false;
        }

        srcOffset = static_cast<uint32_t>(srcRow) * blockBytes;
        dstOffset = static_cast<uint32_t>(dstRow) * blockBytes;
        return true;
    }

    /**
     * @brief 从 GM 拷贝一行到 UB
     * @param srcOffset 源缓冲区中的字节偏移
     *
     * 从拷贝队列分配缓冲，执行 GM 到 UB 的 DataCopy，
     * 并将缓冲入队以供后续 copy-out。
     *
     * @note 假设分配的 tensor 中有 blockBytes 字节可用
     * @note DataCopy 执行 DMA 传输（硬件上异步）
     */
    __aicore__ inline void CopyIn(uint32_t srcOffset)
    {
        AscendC::LocalTensor<CopyUnit> local = copyQue.AllocTensor<CopyUnit>();
        AscendC::DataCopyExtParams copyParams{1, blockBytes, 0, 0, 0};
        AscendC::DataCopyPadExtParams<CopyUnit> padParams{false, 0, 0, 0};
        AscendC::DataCopyPad(local, srcGm[srcOffset], copyParams, padParams);
        copyQue.EnQue(local);
    }

    /**
     * @brief 从 UB 拷贝一行到 GM
     * @param dstOffset 目标缓冲区中的字节偏移
     *
     * 从拷贝队列出队缓冲，执行 UB 到 GM 的 DataCopy，
     * 并将缓冲释放回队列。
     *
     * @note 假设缓冲包含 blockBytes 字节的有效数据
     * @note DataCopy 执行 DMA 传输（硬件上异步）
     */
    __aicore__ inline void CopyOut(uint32_t dstOffset)
    {
        AscendC::LocalTensor<CopyUnit> local = copyQue.DeQue<CopyUnit>();
        AscendC::DataCopyExtParams copyParams{1, blockBytes, 0, 0, 0};
        AscendC::DataCopyPad(dstGm[dstOffset], local, copyParams);
        copyQue.FreeTensor(local);
    }

private:
    AscendC::TPipe *pipe = nullptr;
    AscendC::TQueBind<AscendC::TPosition::VECIN, AscendC::TPosition::VECOUT, BUFFER_NUM> copyQue;

    AscendC::GlobalTensor<CopyUnit> srcGm;
    AscendC::GlobalTensor<CopyUnit> dstGm;
    AscendC::GlobalTensor<int64_t> srcIndexGm;
    AscendC::GlobalTensor<int64_t> dstIndexGm;
    AscendC::GlobalTensor<uint8_t> validMaskGm;

    uint32_t srcRows = 0;
    uint32_t dstRows = 0;
    uint32_t blockBytes = 0;
    uint32_t alignedBlockBytes = 0;
    uint32_t maxCopy = 0;
    uint32_t copyRowsPerCore = 0;
};

/**
 * @brief host 通过 aclrtLaunch 调用的内核入口点
 *
 * @param src 源缓冲区（GM 地址）
 * @param dst 目标缓冲区（GM 地址）
 * @param src_index 源行索引（GM 地址，int64_t 数组）
 * @param dst_index 目标行索引（GM 地址，int64_t 数组）
 * @param valid_mask 有效性掩码（GM 地址，uint8_t/bool 数组）
 * @param srcRows src 中的总行数
 * @param dstRows dst 中的总行数
 * @param blockBytes 每行字节数
 * @param maxCopy 映射条目数量
 *
 * 启动配置：
 * - block_dim：使用的 AI Core 块数（通常 8-48）
 * - stream：用于异步执行的 ACL 运行时流
 *
 * @note 这是 host（Python/C++）和 device（AI Core）之间的 ABI 边界
 */
extern "C" __global__ __aicore__ void unidex_copy(GM_ADDR src, GM_ADDR dst, GM_ADDR src_index, GM_ADDR dst_index,
                                                  GM_ADDR valid_mask, uint32_t srcRows, uint32_t dstRows,
                                                  uint32_t blockBytes, uint32_t maxCopy)
{
    AscendC::TPipe pipe;
    KernelUniDexCopy kernel;
    kernel.Init(src, dst, src_index, dst_index, valid_mask, srcRows, dstRows, blockBytes, maxCopy, &pipe);
    kernel.Process();
}
