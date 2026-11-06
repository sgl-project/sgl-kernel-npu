# key_pool

公共入口：`from sgl_kernel_npu.operators.mem_cache import key_pool`。
`auto/aclnn` 均使用独立 attentions 扩展，调用
`aclnnKeyPoolGetWorkspaceSize` 和 `aclnnKeyPool`；不增加 cann_ops_transformer
Python 依赖，不自动回退。

参数和例子见[英文接口说明](../../../en/operators/mem_cache/key_pool.md)。hidden_states
为连续 ND BF16/FP16 TH 或 BSH，wk/gate_weight 为同 dtype 的 [D,H]，APE 为
FP32 [r,D]。state_cache 为 FP32 [blocks,block_size,2D]，前 D 列保存归一化
Key、后 D 列保存 Gate；仅第 0 轴允许额外 stride。schema 使用 Tensor(a!) 标注原地写。

cache_block_table 和 start_pos 为 NPU int32，请求级表映射中间 state，不能传最终
index 页表。TH 的 cu_seqlens 为含首零、末尾 T 的 NPU int32 [B+1]，BSH 不传；
不转换 host list。norm_weight/bias 为成对 FP32 [D]，cos/sin/seqused 必须为 None。

单 Tensor 返回。TH 分配 min(T,T//r+B) 行，仅前面按请求连续拼接的有效行可读；
每请求有效数量为 (start+Q)//r-start//r。其余行未初始化。接受长度不影响算子计算，
由调用方单独约束 index 提交，不能改变后一请求的 compact 输出偏移。

部署必须重建并安装 attentions wheel，主 wheel 不包含它：

```bash
bash build.sh -a attentions
python3 -m pip install --no-deps --force-reinstall output/attentions-*.whl
python3 -c 'import torch, torch_npu, attentions; print(torch.ops.attentions.key_pool.default._schema)'
python3 -m pytest -q tests/python/operators/mem_cache/test_key_pool.py
```

实际 CANN/custom-op 环境须导出两个 KeyPool 符号。schema 导入不等于 C++ 编译、
数值或 NPUGraph 验收；9.2.0 接口未提供 TorchAir converter。
契约来源：[官方固定版本](https://gitcode.com/cann/ops-transformer/blob/9.2.0/attention/key_pool/docs/torchapi_key_pool.md)。
