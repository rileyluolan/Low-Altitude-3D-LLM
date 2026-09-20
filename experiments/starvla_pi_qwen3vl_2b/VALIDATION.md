# 导出副本校验记录

校验日期：2026-09-17 UTC。检查在独立发布副本中完成，使用现有 Python 环境及共享的模型/数据输入；未启动训练或 GPU 评测。

## 已通过

- 在干净的固定版本 StarVLA / HUGE-Bench checkout 上应用三份补丁，并检查反向应用。StarVLA 六个改动文件的内容 SHA256 与当前运行实验完全一致。
- Python 语法和 Bash `bash -n` 检查。
- OXE init、align、joint、HUGE base、train、smoke 六份配置与历史/当前原文件逐项比较；除路径模板和明确补入的 OXE 累积步数外，配方一致。
- 无 GPU dry run：HUGE 正式训练为 4 卡 / batch 512 / 16,798 步；smoke 为 4 卡 / batch 8 / 2 步；OXE 对齐为 8 卡 / batch 128 / 5,000 步；OXE 四卡联合为 batch 128 / 50,000 步。
- CPU 权重迁移测试：36→28 动作层选择、保留 2B backbone 和新 projector；7D→4D 的两个 encoder、decoder 和 bias 通道；意外缺失/多余张量触发错误。
- 八个构建、数据准备、策略服务、rollout、阶段核验 CLI 的 `--help` 导入检查成功。
- 在发布副本中建立真实 HUGE 数据视图和索引，识别 5,175 条训练轨迹、1,720,096 帧；检查图像、状态、动作 schema 和注册器。
- 共享的现有 HUGE base 通过结构与完整文件 SHA256 校验：1,154 张量，四维输入/输出，28 层，SHA256 为 `87e38d987fd2d1b7fd50b77529717d4fc19111b1c0ca58096f24e933b79b6c4b`。
- tmux 启动器用无 GPU 测试命令实测，退出码为 0，日志保留 `\r` 进度刷新。测试会话已移除；正在训练的会话未修改。

CPU 迁移测试可重复执行：

```bash
source scripts/lib/common.sh
CUDA_VISIBLE_DEVICES='' "$VENV_ROOT/bin/python" -m unittest discover -s tests -v
```

## 尚未在发布副本重跑

- 全新环境安装、完整 OXE/HUGE 数据下载。
- 使用全部真实源权重重建初始化，以及 5k+50k OXE 训练。
- 完整五 epoch HUGE 训练和 993 条闭环测试轨迹。

因此这些检查证明代码衔接、张量迁移规则与当前实验配置可核对，并不等同于全流程训练完成或测试性能已验证。历史 base 的 OXE 步数来源于已有 manifest，原始 OXE 训练日志本次未取得。


## 2026-09-20：已完成结果归档

以上为 2026-09-17 发布时的代码验证范围。原本机实验随后完成了 16,798 步训练和 993 条正式闭环测试；结果见 [完整结果](records/hugebench_5ep_b512/README.md)。本次归档逐一核对 18 份原始文件 SHA256，训练步数连续且 loss 有限，993 条逐轨迹指标与官方 seen/unseen 均值一致。图表从完整记录重新生成，视频完成记录为 993 条、339,394 帧。此次未在发布 checkout 重跑训练或闭环评测。
