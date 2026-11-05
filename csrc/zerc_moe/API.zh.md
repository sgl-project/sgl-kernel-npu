# ZeRCMoE 源码和构建

贡献者：chengyue、zhaoyuxin、hanggongmao、chenyukai。

`source/` 下的源码来自 [ops-transformer](https://gitcode.com/hanggongmao1/ops-transformer)
的 `mega_moe` 分支提交 `d4c5f65d93591f26895c126fc590f0d622440819`。
原版权声明和 CANN Open Software License Agreement 2.0 均予保留，见
[LICENSE](LICENSE)。仓库根目录的 MIT 许可不取代这些文件的原许可。

导入算子使用独立的 CANN 和 Torch 名称，以便与现有 MegaMoe 实现共存。
公开接口、构建选项、输入约束和通信缓冲区生命周期见
[zerc_moe](../../docs/api/zh/operators/moe/zerc_moe.md)。

构建、设备运行和运行时共存尚未验证。可选的标准构建使用固定版本的
ops-transformer MC2 构建框架；独立命令 `python3 csrc/zerc_moe/build.py --plan`
只展示对应构建步骤，不会编译或安装。
独立 wheel 沿用上游的 `npu_ops_transformer` 包名。不要在同一进程中同时
导入该 wheel 和标准 wheel 内置的 ZeRCMoE 绑定；两者会注册同一个
`zerc_moe` Torch 命名空间。
