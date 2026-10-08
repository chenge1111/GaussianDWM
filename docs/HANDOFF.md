# 给 H100 实验执行者

项目入口：README。代码状态：实现交付；未执行测试、训练或 GPU 验证；无已报告实验数值。

公开数据未转换/未匹配时，先执行 [PUBLIC_DATA_TASK](PUBLIC_DATA_TASK.md)，其中包括下载入口、具体命令、Gaussian 编号映射所需证据和应回传的报告。在线路线不依赖作者完整离线高斯；离线 QA 可以不读 LiDAR，但不能省略确认的 Gaussian 帧身份和坐标变换。

## 需要准备

1. 有授权的 `dtc111/GaussianDWM` 基础权重，固定 revision 或同一份本地下载目录。
2. nuScenes trainval、真实 QA 标注与训练/验证场景划分。
3. 在线联合配置所需当前/未来深度、CLIP/SAM 教师、GT 轨迹；工具在 prepare_nuscenes/prepare_semantics。
4. 可选 DrivingForward SF 初始化权重，并记录使用与否。
5. H100 CUDA toolkit、编译器、集群环境；脚本默认单节点8卡，不表示已测最佳资源设置。

## 入口

- 安装：`bash scripts/setup_h100.sh`。
- 训练：设置 `MANIFEST, DATA_ROOT, OUTPUT_DIR, CONFIG, GPUS_PER_NODE`，运行 `bash scripts/train_h100.sh`。
- Slurm：按集群修改 `scripts/train_h100.sbatch` 账户/分区/GPU规格。
- 推理：`python -m gaussiandwm_research.infer --checkpoint .../final --manifest ... --data-root ... --output-dir ...`。
- 数据/指标：README 和 DATA_CONTRACT。

## 配置与报告应一致

固定相同基础权重、数据划分、seed、步数和监督；记录所有 coarse/fine/resample 调用成本，而不是只计算最后一次2048/4096 Gaussian tokens。在线与离线实验必须注明 Gaussian 来源和重建初始化；无实例 ids 的语义补全分支不能报作完整同实例增强。

正式结论由实验执行者验证，代码交付没有证明论文指标、遮挡召回、实时性能或新颖性。官方 QA/FVD 协议需要合作者接入，不使用近似指标替换官方指标。
