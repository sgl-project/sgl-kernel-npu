# BF16 布局与转换片段

在 dav-3510 的 SIMD VF 中处理连续 BF16 数据时查阅。片段提取自已验证的有限值计算路径，工具链记录见[SDK 读码笔记](../cann-cce-intrinsics.md)。以下是局部片段，不是独立 Kernel，尚未在本次整理后单独编译或上板验证。

## 整数拆分

每次读取 128 个 BF16，视为 64 个 u32；低半对应偶数元素，高半对应奇数元素。`highmask` 的每个 u32 lane 为 `0xffff0000`，`m` 为全 64 个 B32 lane。需要完整 256B 可读 UB 区域、满足 NORM 装载对齐；尾块先搬入对齐缓冲并补零，不能靠计算 mask 防止整寄存器读取越界。

```cpp
__simd_callee__ inline void bf_unpack(__ubuf__ uint16_t* src,
    vector_f32& even,vector_f32& odd,vector_u32& highmask,vector_bool& m) {
    vector_u32 p,a,b;
    vlds(p,(__ubuf__ uint32_t*)src,0,__cce_simd::NORM);
    vshls(a,p,16,m);vand(b,p,highmask,m);
    even=(vector_f32&)a;odd=(vector_f32&)b;
}
```

## FP32 有限值 RNE 打包

以下位于 VF 内，`lo`、`hi` 为 64 lane 的偶/奇 FP32 结果。`roundbias`、`one`、`highmask` 分别广播 `0x7fff`、`1`、`0xffff0000`；`output` 是足够大的 UB uint16_t 缓冲，`row*stride+col` 指向当前 128 元素块。输出须按搬运有效范围处理尾块。

```cpp
            vector_u32 l=(vector_u32&)lo,h=(vector_u32&)hi,lsb,hsb,packed;
            vshrs(lsb,l,16,m);vand(lsb,lsb,one,m);
            vshrs(hsb,h,16,m);vand(hsb,hsb,one,m);
            vadd(l,l,roundbias,m);vadd(l,l,lsb,m);vshrs(l,l,16,m);
            vadd(h,h,roundbias,m);vadd(h,h,hsb,m);vand(h,h,highmask,m);
            vor(packed,l,h,m);  // codespell:ignore vor
            vsts(packed,(__ubuf__ uint32_t*)(output+row*stride+col),0,__cce_simd::NORM_B32,m);
```

位重解释不做数值整型转换。舍入通过加 `0x7fff + ((bits>>16)&1)` 后保留高 16 位实现，偶/奇结果组成一个 u32。不要对 NaN payload、无穷或接近溢出的值直接承诺与框架完全一致，这些需要单独定义并测试。

## 验证边界

用 lane 编号区分偶奇、正负值和 RNE 中点，测试往返布局、尾块及输出哨兵。若需要逐位一致，保持原 FP32 乘加顺序，不能同时替换为 FMA。历史某种 PART_ODD 调用失败不能推导为 BF16 转换硬件普遍不可用；其他 `vcvt` 路径必须按目标重载的输入布局重新验证。
