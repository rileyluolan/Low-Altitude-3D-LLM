# 发布验证

2026-09-20，将本机实验导出为可配置路径版本；活动训练目录、模型权重与进程保持原样。

## 本次检查

- Python AST、shell 语法、JSON/YAML 解析通过。
- 在独立、干净的固定版本 StarVLA checkout 上应用补丁；六个改动模块与活动实验逐字节相同。
- 正式配置解析后与活动实验的全部非路径超参数一致；四卡 batch 512、累积 16、16,798 次更新。
- 正式和 smoke 的 dry run 通过，Accelerate/DeepSpeed 配置写入独立 `runtime/`。
- 发布下载器用小型模拟 HTTP 数据验证完整下载、Range 续传以及错误 SHA256 拒绝逻辑。
- 新旧实际 tokenizer 的 BPE merges 经序列化格式归一后完全相同。
- 评测、视频生成命令的参数解析通过；评测使用 `MODEL_ROOT` 的 RefDrone 兼容 VLM 路径。
- tmux 启动器用无 GPU 命令检查输出和退出状态；正式训练进程未改动。

## 检查范围

真实 GPU 前向、反向、626 个 VLM 张量的一致性及动作输出检查来自
[2026-09-19 原始 preflight](records/hugebench_refdrone_5ep_b512/preflight.json)。
本次未重新组装数 GB 权重、安装整套环境、启动额外训练或进行闭环评测。
发布版对原组装过程增加输入 SHA256 固定、BPE merges 和输出目录检查；这些保护不改变训练张量。

原实验最终训练和评测结果的归档检查见
[结果说明](../starvla_pi_qwen3vl_2b/records/hugebench_5ep_b512/README.md)。
