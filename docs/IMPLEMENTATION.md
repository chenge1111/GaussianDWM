# 实现位置与梯度路径

## 采样路线

`cascade.py` 用 view-stratified uniform 取得 1024 全局 anchors。`bridge.py` 将动态长度 Gaussian embeddings 写入原 Qwen Gaussian placeholders。`inference.py` 让 Qwen 输出 coarse JSON，再执行任务相关精采样。元素配额最多占预算 80%，其余由全局 score 填充；配额重叠或目标候选不足时实际选择不重复的可用点。

score 是原 CLIP cosine 与新增 projected query-key dot product 的和，query 同时由问题 CLIP 与全局 Qwen 隐状态决定。遮挡项加 `log(2)`，因此正注意力概率权重乘 2；直接乘负 cosine 会错误压低遮挡目标，未采用这种做法。

局部任务对 current-ego 3D bounds 限制候选。缺失 bounds 时保留全局恢复；空区域通过 telemetry 标记，不宣称覆盖了目标。反馈阶段解析 `[RESAMPLE]` JSON，扩大搜索区域并重新采样，最多两个重试；不使用无限循环来保证无法保证的目标命中。

`occlusion.py` 用投影深度 z-buffer 与九点高斯 footprint 比较。被可见的任一相机清楚看到的点不会当完全遮挡；不透明度只用于建立可信前景深度，不作为遮挡标签。覆盖是近似几何估计，阈值 0.5 不等于有 ground-truth 遮挡标签。

语义补全只从**可见、已知同实例**邻居取均值。完全没有可见同实例 donor 时保持原语言值；unknown=-1 不合并。图像 SAM mask 内 identity 和真正跨视角物体 identity 的差异要在实验解释。

## 在线网络

`online.py` 调用 DrivingForward 实际 `DepthNetwork` 和 `GaussianNetwork`，采用 SF 六视角模式。官方网络源码在 `vendor/drivingforward`，使用现代 torchvision ResNet 和 Rodrigues compatibility layer 替换外部 PackNet/PyTorch3D imports。保留官方网络参数 key，支持 `depth_net.pth/gs_net.pth` 初始化。

新增语言、motion、RGB residual、scale residual 和 offset heads；语言值保持 CLIP512，并用训练的 512→3 compressor 接入原 Gaussian aligner，另有低幅语义残差映射。Gaussian quaternions 从相机系转换到当前 ego，静态/动态用 motion probability 混合，未来位置 `xyz + velocity * probability * dt`。

这属于 DrivingForward 适配与新增学生分支，**不是 CVD-STORM，也不是声称复现 DrivingForward 所有原渲染/训练细节**。当前颜色采用输入 RGB 与学生残差，未使用完整视角相关 SH 渲染。高斯 stride 默认8，可改4提高密度，需要重新衡量显存。

## 联合损失

`model.py`：

- 当前及真未来可微渲染的 RGB L1、有效稀疏深度 L1。
- 教师语言场与渲染语言场余弦损失，以及 CLIP512→3→512 编码一致性。
- Coarse JSON（有监督时）和 QA assistant-only 交叉熵。
- 保留发布版 EDM denoiser 损失，另加预测去噪 latent 经冻结 VAE **有梯度解码**后的 RGB MSE、深度 L1 与 VGG 感知损失。
- 真实轨迹 SmoothL1、真实/轨迹派生指令 CE、几何风险代理、生成风险 BCE、风险反馈后轨迹/指令监督。

QA generation condition 只池化 assistant 答案之前的 causal hidden states，避免使用 teacher-forced GT 答案作为未来条件。未来 RGB/depth 仅作目标，不作为当前场景或伪布局条件输入。训练语义 mask 来自教师，不能通过降低预测 opacity 逃避语言监督。

连续梯度链：future loss → denoiser/condition fusion → Qwen hidden / predicted plan / differentiable layout encoder → gsplat renderer / sampled Gaussian values → geometry / language / motion heads → DrivingForward。冻结 VAE/CLIP 权重仍可对输入 latent/条件保留梯度；只对 GT 编码和教师特征采用 no_grad。

推理的 top-k、预算整数、实例匹配与 prompt JSON 为离散操作。训练用每个 anchor 的最多32空间邻居 soft attention 作连续松弛，梯度通过 score、几何距离与 Gaussian values；不会声称对硬 top-k 索引或文字反馈全局精确可微。

`online_no_joint.yaml` 保留独立重建损失，detach 场景后才进入理解/生成；用于隔离下游联合反传作用。

## 原始代码的最小改动

- `qwen_gauss_backbone.py`：增加 `selected_gauss_hidden` 外部动态 token 路径，原 similarity/identity 分支保持。
- `world_head.py`：EDM 输出补充 denoised_latents，使研究模块可进行保留梯度的 VAE 解码。
- `gauss_normalizer.py`：兼容 NumPy1.26/2.x 的 core 路径，便于 nuScenes SDK 共存。

原 `gaussiandwm_cvpr.scripts.train_qa/train_world` 仍是发布版入口；它们没有被称为已修复多卡。新的研究训练器在 `gaussiandwm_research.train`，接入 Accelerate/DeepSpeed、梯度累积、精度和恢复。

## 训练/推理制品

Accelerate checkpoints 保存完整训练状态；portable `final/` 保存可训练参数及 online buffers 和配置、processor。下载原基础权重并重建相同 LoRA 结构后加载增量；它不是无需基础权重的独立模型。

本项目只实现代码，未执行验证。缺少训练的参数不会凭空具有目标能力；需在真实数据训练后再评估。FID wrapper 与原近似 QA evaluator 不构成论文完整评测协议，FVD 仍需外部官方协议。
