# 数据契约

JSONL 每行一个训练/推理样本。禁止将样例或伪造答案当实验数据。图片路径可相对 `--data-root`，所有缓存路径建议使用集群共享绝对路径。

## 必须字段

| 字段 | 类型/约定 |
|---|---|
| sample_uid | 唯一字符串，QA 与场景一对多时不要重复 |
| query / answer | 真实查询/答案；推理可省略 answer |
| task_kind | `global` 或 `local`，默认 global |
| image_paths | 六个 RGB 路径，按 CAM_FRONT, FRONT_LEFT, FRONT_RIGHT, BACK_LEFT, BACK_RIGHT, BACK |
| intrinsics | `[6,3,3]`，原始图像像素坐标；loader 在 resize 时自动缩放 |
| camera_to_ego | `[6,4,4]`，camera → **当前参考 ego frame**，含相机时间戳的 ego pose 修正 |
| clip_text_feature_path | `[512]` `.npy/.npz/.pt`，CLIP ViT-B/32 查询特征 |

ego 坐标：x 前，y 左，z 上；位置/深度单位米；四元数 wxyz。Gaussian 的尺度使用 log(scale)，opacity 使用激活值。相机是 OpenCV x 右/y 下/z 前，通过 camera_to_ego 转换。

## 在线联合训练监督

| 字段 | 类型/约定 |
|---|---|
| depth_paths | 6 个二维稀疏 metric-depth `.npz/.npy`；0 表示无 GT，不能补为假 GT |
| semantic_feature_paths | 6 个 `[512,Hs,Ws]` 教师特征；默认 44×80，可用更高分辨率保留小目标 |
| instance_id_paths | 6 个 `[H,W]` 或教师网格二维实例标识；-1 未知；整数需避免视角间冲突 |
| future_image_paths | `[T,6]` 真未来 RGB 路径，默认 T=6，0.5 秒间隔 |
| future_depth_paths | `[T,6]` 真未来 metric-depth 路径，不能使用模型生成深度充当 GT |
| future_camera_to_ego | `[T,6,4,4]`，都位于**当前**ego坐标，不是各时刻各自 ego 坐标 |
| trajectory | `[T,5]`，当前 ego 下 `[x,y,z,sin(yaw),cos(yaw)]`，yaw 相对当前车身 |
| command | brake/straight/left/right；转换工具的默认标签由真实轨迹阈值派生，非人工驾驶指令 |
| scene_hint | 可选 `{summary,complexity,elements:[{name,weight,bounds}],target_bounds}`；训练粗认知监督 |

粗阶段 GT hint 仅在训练读取。推理不使用 manifest 中的 hint、未来轨迹或未来图像，而是从当前图像和当前高斯生成粗认知。`prepare_nuscenes` 使用 nuScenes GT 框制作 coarse 监督，不代表当前输入观测到了每个 GT 对象。

默认联合配置需要未来 RGB/depth；重建深度、语言场和轨迹监督若缺失，相关损失无法生效。建议提供完整这些字段；不要把缺失监督下的训练命名为已完成全部联合目标。

## 离线 QA 的额外字段

- `gauss_paths`：原 LangSplat `.pth` / `.npz` / packed `.npy` 文件列表，字段 `_xyz,_scaling,_rotation,_opacity,_language_feature`。高斯缩放/opacity 遵循原 normalizer 的语义；其中 opacity 原始为 logits。
- `gauss_to_ego`：可选每个 Gaussian 文件坐标系 → 当前 ego 的 `[4,4]` 变换。省略代表文件本来已经是当前 ego 坐标，不能靠代码猜坐标系。
- `gauss_instance_ids_path`：可选 `[N]` 实例 ids，与全部 Gaussian 拼接顺序一致。**有它才有可靠同实例均值填充**；缺失时只启用遮挡加权，不冒充已完成同实例关联。

离线 Gaussian 的源文件不保证有完整颜色，本项目离线配置只跑 QA，避免将无颜色的占位值作为真实渲染。在线生成使用在线网络的颜色预测。

## 教师与存储

CLIP/SAM 是冻结的监督模型，不参与在线推理。训练通过语义渲染的余弦损失将教师信号投到语言分支。SAM 的 image-local ids 不提供跨相机/时间身份；需要该身份时替换为已关联/标注实例数据。

语义缓存 `[6,512,44,80]` float16 每样本约 21.6 MB（压缩率依数据而定）。共用同一帧的 QA 样本共享图片语义缓存，文本按查询 hash 共享。GT、特征与所有输出应放共享存储，不提交 Git。
