# 模型改造与权重迁移

## 原始模型和目标模型

| 组件 | 官方源 StarVLA-PI | 2B OXE 模型 | HUGE 模型 |
| --- | --- | --- | --- |
| VLM | Qwen3-VL-4B | Qwen3-VL-2B-Instruct | 训练后的同一 2B VLM |
| VLM hidden size | 2560 | 2048 | 2048 |
| layer-wise Action DiT | 36 层、hidden=1024 | 28 层、hidden=1024 | 28 层、hidden=1024 |
| 每层 projector | 2560 → 1024 | 2048 → 1024 | 保留 OXE 训练后的 projector |
| state/action dim | 7 / 7 | 7 / 7 | 4 / 4 |
| action horizon | 16 | 16 | 20 |

精确源码、模型仓库、revision 和源文件 SHA256 均在 `SOURCE.lock.yaml`。上游已经提供通用 `QwenPI_v3` 框架；2B 接入复用了这套逐层交叉注意力实现，而模型实例化配置、源动作层选择、新 projector、两阶段训练和四维迁移由本实验脚本定义。

## 1. 替换 backbone，并保留动作模块的可迁移权重

入口：`scripts/build_oxe_base.sh` → `scripts/pretraining/build_oxe_base_checkpoint.py`。

1. 固定随机种子 42，使用官方 2B VLM 初始化完整 `qwen_vl_interface`。
2. 由 VLM 配置自动确定 28 个语言层和 hidden=2048，构建 28 个 2048→1024 的 LayerNorm+Linear projector。
3. 原 4B 的 projector 形状不兼容，不能原样复制；新 projector 在后续对齐阶段训练。
4. 为每个目标动作 block 选择相近相对深度的源 block：

   ```python
   source_index = round(target_index * 35 / 27)
   ```

   目标 0..27 对应源：

   ```text
   0,1,3,4,5,6,8,9,10,12,13,14,16,17,18,19,21,22,23,25,26,27,29,30,31,32,34,35
   ```

5. 复制全部形状兼容的动作编码器、状态编码器、28 个选定 DiT block 和解码器；目标动作模块应有 **416/416 张量被迁移**。发现缺失或意外维度不匹配时直接报错，不静默随机初始化动作权重。
6. 保存权重、完整配置、源数据归一化和逐张量迁移 manifest。

## 2. OXE 适配

先训练 5,000 步 projector，将新 2B 表征接到原有动作网络；这时 VLM 和动作模块均冻结。随后全部解冻，在 Bridge+RT-1 上联合训练 50,000 步。

`prepare_oxe.py` 使用源 VLA 配套的 OXE 归一化统计，按非空语言轨迹建立 step 缓存。`datasets.py` 的可选补丁让 trajectory sampling 与这个有效样本集合保持一致。完整配方包括固定单路相机、CoT 提示词、0.5/0.5 数据混合、学习率、warmup、噪声参数和扩散重复数。

## 3. HUGE 四维迁移

入口：`scripts/pretraining/convert_mature_oxe_to_huge.py`；实际张量操作集中在 `scripts/tools/checkpoint_transfer.py`。

源通道 `[x,y,z,roll,pitch,yaw,gripper]` 中选 `[0,1,2,5]`，形成四维初始化。变换仅涉及：

| 张量 | 操作 |
| --- | --- |
| `action_model.action_encoder.layer1.weight` | 保留输入列 `[0,1,2,5]` |
| `action_model.state_encoder.layer1.weight` | 保留输入列 `[0,1,2,5]` |
| `action_model.action_decoder.layer2.weight` | 保留输出行 `[0,1,2,5]` |
| `action_model.action_decoder.layer2.bias` | 保留元素 `[0,1,2,5]` |

其余张量必须原样迁移；转换不得出现随机初始化张量。Horizon 从 16 配置为 20，没有新增按这两个长度区分的训练参数。转换后使用 HUGE 训练集的四维 state/action 统计，来源 revision 和哈希见记录文件。

这个通道映射提供可训练的初始化，**不代表机器人末端与无人机的状态、单位、坐标系和动力学天然相同**。HUGE 数据适配器定义绝对状态 `[x,y,z,yaw]` 与已存储的增量动作 `[dx,dy,dz,dyaw]`，由下游训练完成语义适配。使用四维头，无需额外 padding。

## 源码补丁的职责

- `0001-starvla-qwen2b-oxe.patch`：Qwen3 梯度检查点和 logits 内存控制；有效语言轨迹筛选。
- `0002-starvla-training-runtime.patch`：Accelerate/DeepSpeed 累积步数一致、只在优化器更新后记录/保存、跨 microbatch/rank loss 平均、可选 shuffled frame passes、严格加载完整权重、推理优先读取完整配置。
- `0003-hugebench-lazy-openpi.patch`：把仅 OpenPI 推理需要的导入移到官方主入口，允许本实验直接复用官方 rollout 工具函数。

第二份补丁包含当前四卡训练的运行修正。历史 8 卡 OXE 配方与本次发布的可移植运行脚本有区别；提交包含原配方和迁移语义，不承诺相同训练硬件、软件版本以外的逐位一致性。

## 当前已有 base 的核验结果

记录中的 HUGE base 共 2,608,557,060 个参数、1,154 个张量，28 个 projector 与 28 个动作 block。其 SHA256 为 `87e38d987fd2d1b7fd50b77529717d4fc19111b1c0ca58096f24e933b79b6c4b`。

此前 CPU 审计确认所有张量有限；与官方 2B VLM 对比，625 个 VLM 张量中 593 个发生变化，28 个语言层均有更新。该证据支持 VLM 已经过适配训练，但本身不能证明历史训练准确完成了多少步；5k+50k 的历史步数依据转换/训练记录，而本次尚未取得原始 OXE 日志。
