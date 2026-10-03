# GaussianDWM：遮挡感知级联采样与在线联合训练

在 [GaussianDWM CVPR 开源代码](https://github.com/dtc111111/GaussianDWM)上实现两条研究路线：

1. **遮挡感知的级联任务采样**：1024-token 全局认知、2048–4096 自适应精采样、区域重采样与同实例语义增强。
2. **在线 GaussianDWM**：六视角图像 → DrivingForward 前馈高斯 → 语言场 → 理解/规划/RGB-D 生成 → 风险反馈与规划修正。

本仓库包含模型、数据准备、训练、推理、消融配置和 H100 集群启动脚本。**按项目负责人的要求，代码未运行测试、训练或 GPU 验证；目前没有实验结果，不保证运行兼容性、指标增益或实时速度。** 数据集和预训练权重需单独准备。

## 实现内容

| 模块 | 实现 |
|---|---|
| 粗采样 | 对有效高斯按视角均匀抽样，默认最多 1024，不按低不透明度删除目标 |
| 全局认知 | 使用原 Qwen 输出结构化场景概述、元素语义权重、ego 坐标范围和复杂度 |
| 精采样 | 保留原 CLIP 相似度并增加任务/全局条件 cross-attention 打分；全局元素配额、局部空间筛选 |
| Token 预算 | 默认在 2048–4096 间按复杂度调整；候选不足时使用实际数量，不复制 token 凑数 |
| 遮挡增强 | 投影深度和九点 footprint 覆盖估计；遮挡注意力权重 ×2；已知同实例可见邻居语言特征均值补全 |
| 反馈采样 | 严格解析 `[RESAMPLE]` 区域请求，默认最多两次；记录空区域、无效请求和耗尽状态 |
| 在线重建 | 实际接入 DrivingForward 的 DepthNetwork / GaussianNetwork 源码；现代 PyTorch 兼容层 |
| 语言场与运动 | 联合输出 512 维 CLIP 语言特征、颜色残差、几何偏移、运动概率和速度 |
| 可微训练 | 空间邻域 soft attention、gsplat 渲染与保留梯度的 VAE 解码连接重建、QA、生成损失 |
| 规划反馈 | 从 Qwen 隐状态预测轨迹/指令，轨迹控制未来视角和生成，生成风险指导轨迹/指令修正 |
| 集群 | Accelerate + DeepSpeed ZeRO-2、BF16、梯度累积、断点续训、单机/多机 H100/Slurm 脚本 |

原公开采样器是 CLIP similarity/identity，未包含用户设想中的交叉注意力采样模块。因此新增的 cross-attention scorer 是本项目实现，不能称为直接复用了原仓库中不存在的模块。原 Qwen、Gaussian aligner、条件融合、UNet、VAE 和原始入口保留。原始说明见 [UPSTREAM_README](docs/UPSTREAM_README.md)。

## 安装（由实验执行者在集群操作）

推荐独立 Python 3.11 环境、H100、支持 CUDA 12.6 的驱动和 CUDA toolkit / C++ 编译器。gsplat 需要 CUDA 编译环境。脚本是环境起点，尚未在目标集群运行。

```bash
git clone https://github.com/chenge1111/GaussianDWM.git
cd GaussianDWM
conda create -n gdwm-online python=3.11 -y
conda activate gdwm-online
bash scripts/setup_h100.sh
hf auth login
```

基础权重为 [`dtc111/GaussianDWM`](https://huggingface.co/dtc111/GaussianDWM)，配置固定到 `a99358a60f54386b0bee9c4d3fa5a3fe058a2514`，需要获得该 gated 模型的访问权限。可将 `base.model_id` 改为已下载的本地目录。

DrivingForward 初始化权重可选：下载官方 SF 权重，将 `reconstruction.weights_dir` 指向包含 `depth_net.pth` 和 `gs_net.pth` 的目录。不提供时采用 ImageNet ResNet 初始化及新初始化的其余重建参数；这会改变训练难度，报告实验时需注明。

## 准备真实数据

本项目不自带 nuScenes、完整 QA 标注或训练权重。先准备 nuScenes trainval，整理真实 QA 为 JSONL，最低字段是 `sample_token`、`query`、`answer`，可提供 `task_kind: local/global` 及 `scene_hint`。

```bash
python -m gaussiandwm_research.prepare_nuscenes \
  --nuscenes-root /data/nuscenes --qa-jsonl /data/qa/train.jsonl \
  --split train --output /data/gdwm/train.raw.jsonl --depth-cache /data/gdwm/depth

python -m gaussiandwm_research.prepare_semantics \
  --manifest /data/gdwm/train.raw.jsonl --data-root /data/nuscenes \
  --output-manifest /data/gdwm/train.jsonl --cache-root /data/gdwm/semantic
```

第一步从标定/ego pose 和 LiDAR 生成稀疏深度、未来六帧及轨迹，并从真实 nuScenes 框构造**训练用**粗阶段 JSON 监督。验证数据用相同工具的 `--split val`，不能把训练/验证场景混在一起。

第二步使用冻结 CLIP/SAM 制作语言场监督与查询特征。冻结教师不妨碍学生重建网络经渲染损失端到端反传。SAM 标识默认只在单张图像内一致；跨视角/跨时间同实例补全必须提供一致的实例标识，不能把未知实例都视作一个实例。只准备离线 QA 查询特征可加 `--text-only`。

数据细节见 [DATA_CONTRACT](docs/DATA_CONTRACT.md)。离线实验额外需要原 LangSplat Gaussian 文件及明确的坐标转换。

## 训练与消融

在线联合训练：

```bash
export MANIFEST=/data/gdwm/train.jsonl
export DATA_ROOT=/data/nuscenes
export OUTPUT_DIR=/experiments/gdwm/online_joint
export GPUS_PER_NODE=8
bash scripts/train_h100.sh
```

训练卡数和资源消耗需要合作者实测，默认 8 卡只是启动配置。多机每个节点设置 `NODES`、`NODE_RANK`、`MASTER_ADDR` 和一致的端口，或使用 `scripts/train_h100.sbatch`。修改 Slurm 分区/账户/GPU 申请为自己的集群要求。

| 配置 | 用途 |
|---|---|
| `configs/research/offline_similarity.yaml` | 固定 4096 相似度采样参考；原始 CVPR 入口仍独立保留 |
| `configs/research/offline_cascade.yaml` | 改进一：离线高斯级联采样与遮挡增强 |
| `configs/research/online_joint.yaml` | 改进一 + 改进二：完整在线联合训练路线 |
| `configs/research/online_no_joint.yaml` | 阻断下游损失到重建器的梯度，仅重建损失训练重建器 |
| `configs/research/ablate_no_occlusion.yaml` | 关闭遮挡加权与语义补全 |
| `configs/research/ablate_fixed_budget.yaml` | 保留级联流程，固定精采样 4096 |
| `configs/research/ablate_no_feedback.yaml` | 推理时关闭重采样反馈 |

切换方案设置 `CONFIG=configs/research/offline_cascade.yaml` 等。精确续训设置 `RESUME=/experiments/.../checkpoint-500`。`checkpoint-*` 是 Accelerate 训练状态，`final/` 是便于推理的研究增量权重；后者需要同一基础模型，不是独立完整基础权重包。

## 推理与结果导出

```bash
python -m gaussiandwm_research.infer \
  --checkpoint /experiments/gdwm/online_joint/final \
  --manifest /data/gdwm/val.jsonl --data-root /data/nuscenes \
  --output-dir /experiments/gdwm/online_joint/infer

python -m gaussiandwm_research.evaluate \
  --predictions /experiments/gdwm/online_joint/infer/predictions.jsonl \
  --manifest /data/gdwm/val.jsonl --output /experiments/gdwm/online_joint/metrics.json
```

推理导出 QA、轨迹、指令、每次采样预算、实际总 token、重试、风险历史和 RGB/metric-depth NPZ。离线 QA 可加 `--no-world`。规划风险为训练的图像风险预测与几何代理，不是可靠性已获验证的驾驶安全模块。

`evaluate` 提供真实轨迹 ADE/FDE 与 token/延迟统计。`eval_fid` 提供基于 torchmetrics/torch-fidelity 的 FID，但帧选择和分辨率协议需与基线一致。原始 QA 评测为近似指标；官方 grounding、BLEU/ROUGE/CIDEr 和 FVD 协议仍需实验执行者接入，不能将近似指标报作论文原指标。

## 方法边界

- 两阶段 LLM 及重采样增加调用成本；记录全部阶段 token 才能判断效率，不能预先声称“无额外推理负担”。
- 推理使用离散 top-k、JSON 和有上限的反馈循环；训练使用局部连续 soft attention。可微的是连续几何/语言/理解/生成路径，不是离散区域选择和文字反馈本身。
- 正式模型并未训练；“召回率不足 40%”“首次实现”“提升精度”等均不是本仓库已验证的结果。
- 新语言场、速度和规划分支需要真实监督。在线不依赖预重建高斯，不代表无训练、无标定或无数据。
- CVD-STORM 未接入；本项目选择 DrivingForward，使用的源文件和改动已注明。

实现说明见 [IMPLEMENTATION](docs/IMPLEMENTATION.md)，交付与集群入口见 [HANDOFF](docs/HANDOFF.md)。来源/许可声明见 [NOTICE](NOTICE) 及 DrivingForward 随附 MIT license。
