# GaussianDWM 公开数据转换与匹配任务

请在存放真实 nuScenes 和作者公开 Gaussian 的机器上执行以下步骤，产出可追溯的离线 QA 与在线训练清单。优先完成 NuInteract；有 OmniDrive 时再合并。公开标注重整后的数据称为“公开数据重建版本”，不把它报作作者未提供的完全相同处理版本。

任务分工：本仓库提供转换、匹配和报告工具；实验执行者负责实际数据下载、确认 Gaussian 身份和坐标来源、执行转换并回传报告。工具不会凭 `scene_0`、文件排序或最近时间戳猜测对应关系。

## 1 准备输入与目录

需要以下公开资源：

| 资源 | 内容及用途 | 来源 |
|---|---|---|
| NuInteract.zip | train/test PKL，定位、区域描述、预测和规划 QA | [官方下载](https://github.com/zc-zhao/DriveMonkey/releases/download/NuInteract_Dataset/NuInteract.zip) |
| cap_public.tar.gz | 逐帧 caption JSON，包含 token、cam_path、gemini_caption | [官方下载](https://github.com/zc-zhao/DriveMonkey/releases/download/NuInteract_Dataset/cap_public.tar.gz) |
| OmniDrive desc/vqa/conv | 可选逐 sample_token JSON 标注 | [官方 release](https://github.com/NVlabs/OmniDrive/releases/tag/v1.0)、[目录说明](https://github.com/NVlabs/OmniDrive/blob/main/docs/setup.md) |
| nuScenes metadata 和 RGB | 精确帧身份、六视角标定、划分与图像 | [官方安装下载说明](https://github.com/nutonomy/nuscenes-devkit#nuscenes-setup) |
| nuScenes 原始 LiDAR | 仅在制作真实稀疏深度时需要 | 同上；原始 samples/LIDAR_TOP 点云，不是 lidarseg 标签 |
| 作者 Gaussian 压缩包 | 离线基线与采样实验输入 | [GaussianDWM 数据仓库](https://huggingface.co/datasets/xushan/GaussianDWM) |

NuInteract 可以直接读取 ZIP/PKL，无需解压。Caption 可以直接读取 tar.gz。OmniDrive 请解压后仅传入 desc、vqa、conv 子目录，不把模型权重、规划工具或全部 info.pkl 混入 QA 转换输入。保留下载包的版本、下载地址和 SHA256。

```bash
git pull --ff-only
python -m pip install -e '.[data]'
export NUSC_ROOT=/data/nuscenes
export PUBLIC_ROOT=/data/gdwm-public
export PREP_ROOT=/data/gdwm-prepared
export GAUSS_ROOT=/data/gdwm-public/gauss
mkdir -p "$PUBLIC_ROOT" "$PREP_ROOT"

curl -L --fail --retry 3 \
  https://github.com/zc-zhao/DriveMonkey/releases/download/NuInteract_Dataset/NuInteract.zip \
  -o "$PUBLIC_ROOT/NuInteract.zip"
curl -L --fail --retry 3 \
  https://github.com/zc-zhao/DriveMonkey/releases/download/NuInteract_Dataset/cap_public.tar.gz \
  -o "$PUBLIC_ROOT/cap_public.tar.gz"
sha256sum "$PUBLIC_ROOT/NuInteract.zip" "$PUBLIC_ROOT/cap_public.tar.gz" \
  > "$PREP_ROOT/source-sha256.txt"
```

nuScenes 根目录至少包含 v1.0-trainval 的 JSON 表和本次选用帧的六视角 RGB。在线 RGB-D 路线还需当前及未来帧的真实 LiDAR。缺数据时不要制作空白图像、零深度或假答案补齐样本。

## 2 建立 nuScenes 唯一帧索引

```bash
python -m gaussiandwm_research.index_nuscenes \
  --nuscenes-root "$NUSC_ROOT" --version v1.0-trainval \
  --output "$PREP_ROOT/nuscenes.index.json"
```

索引来自 sample、scene、sample_data、sensor、calibrated_sensor 表，记录真实 sample token、scene token、官方 train/val 划分、六视角完整文件名和 sample_data token。场景内 frame_index 按 SDK sample 链从零编号，它**不等于作者 Gaussian 的 frame 编号**。

匹配依据依次使用真实 sample token、keyframe sample_data token、SDK 完整图片路径或 OmniDrive 文件名中的 sample token。多项证据有冲突时拒绝配对；相机 sweep 不自动取最近 keyframe。

使用 mini 时指定 v1.0-mini，后续 prepare_nuscenes 的 split 用 mini_train/mini_val。Mini 仅适合管线和小规模探索，不能报成全量论文实验。

## 3 转换 NuInteract 与可选 OmniDrive

```bash
python -m gaussiandwm_research.convert_public_qa --source nuinteract \
  --annotations "$PUBLIC_ROOT/NuInteract.zip" "$PUBLIC_ROOT/cap_public.tar.gz" \
  --index "$PREP_ROOT/nuscenes.index.json" --output-dir "$PREP_ROOT/nuinteract"
```

转换产生 train.qa.jsonl、val.qa.jsonl、report.json 和 rejected.jsonl。原始文件的 train/test 与官方场景 split 必须一致；NuInteract test 在本项目中记为官方 val。发生冲突时应核查源划分，默认不会把原 train 样本偷偷移入 val。`--source-split-policy sdk-only` 仅适合明确决定重新构建官方场景划分的实验，必须记录这项变化。

3D QA 有时把答案数值放在 assistant turn 的 bboxs_3d_seq，而正文只含 `<embeding>`。转换器会将真实数组写回占位符，并保存 raw_answer 和 grounding_annotation；保留原 9 维框字段，不擅自解释成 7 维 camera/ego 框。坐标变换必须基于原标签的实际定义，不能只因问题出现 CAM_FRONT 就断言原数值是相机系。

2D grounding 默认转为带 camera 的原像素 xyxy JSON，记录 answer_format。问题中的区域坐标不因图像 resize 被自动改写。若需原文字答案，使用 `--grounding-format preserve`；缺数值的 embedding 占位符仍不会被当作有效标签。

NuInteract 的多轮对话默认保留历史，OmniDrive 按公开实现默认拆为独立 QA；可显式指定 `--conversation-policy history` 或 independent。Caption 只从真实发布 caption 建立确定的提问模板，不调用模型编造新的答案。输出中的 source 字段保留文件、记录、对话轮次、配对依据和标签处理方式。

有 OmniDrive 时：

```bash
export OMNI_ROOT=/data/OmniDrive/data/nuscenes
python -m gaussiandwm_research.convert_public_qa --source omnidrive \
  --annotations "$OMNI_ROOT/desc" "$OMNI_ROOT/vqa" "$OMNI_ROOT/conv" \
  --index "$PREP_ROOT/nuscenes.index.json" --output-dir "$PREP_ROOT/omnidrive"
```

来源映射有确凿额外证据时，可传 `--sample-map` JSON：原 token → 真实 sample token，或 `绝对源文件路径#record_id` → 真实 sample token。文件名、编号相似或时间相近不算证据。

## 4 在线路线匹配

在线重建不需要作者预重建 Gaussian。先将 QA 绑定六视角帧并确认图像存在：

```bash
python -m gaussiandwm_research.join_public_data --mode online \
  --qa-jsonl "$PREP_ROOT/nuinteract/train.qa.jsonl" "$PREP_ROOT/nuinteract/val.qa.jsonl" \
  --index "$PREP_ROOT/nuscenes.index.json" --output-dir "$PREP_ROOT/online-matched" \
  --check-images
```

合并 OmniDrive 时，把其 train.qa.jsonl、val.qa.jsonl 同时追加到 --qa-jsonl 参数。重复 QA 会计数去重，训练/验证以 SDK 场景划分为准。

有 LiDAR 时制作联合训练监督：

```bash
for SPLIT in train val; do
  python -m gaussiandwm_research.prepare_nuscenes \
    --nuscenes-root "$NUSC_ROOT" --version v1.0-trainval --split "$SPLIT" \
    --qa-jsonl "$PREP_ROOT/online-matched/$SPLIT.matched.qa.jsonl" \
    --output "$PREP_ROOT/online.$SPLIT.raw.jsonl" \
    --horizon 6 --depth-mode lidar --depth-cache "$PREP_ROOT/depth"
  python -m gaussiandwm_research.prepare_semantics \
    --manifest "$PREP_ROOT/online.$SPLIT.raw.jsonl" --data-root "$NUSC_ROOT" \
    --output-manifest "$PREP_ROOT/online.$SPLIT.jsonl" --cache-root "$PREP_ROOT/semantic"
done
```

未来帧只能从同场景 next 链取，不足六帧的样本进入报告；camera_to_ego、future_camera_to_ego 和轨迹全部相对于当前 CAM_FRONT 时间戳的 ego frame。记录未来真实时间间隔，训练执行者应确认它与配置 step_seconds 一致。

深度输出是**真实 LiDAR 投影的稀疏深度**，零表示无测量。它不是稠密深度 GT。RGB-D 的稠密 latent 监督应保留缺测语义或另设明确的稠密化/教师协议；不得把缺测零值或伪深度说成真实测量。转换工具只负责导出真实测量及其来源。

同一工作可由脚本启动：

```bash
DEPTH_MODE=lidar HORIZON=6 FEATURE_MODE=full bash scripts/prepare_public_data.sh
```

若没有 LiDAR，可运行 DEPTH_MODE=none HORIZON=0 FEATURE_MODE=text 整理当前图像和 QA，但所得清单**不能启动默认在线 RGB-D 联合配置**。补齐 LiDAR 或明确改变监督方案后再使用 world-enabled 配置。

## 5 离线 Gaussian 匹配

先对已解压 Gaussian 做文件清点：

```bash
python -m gaussiandwm_research.match_gaussians \
  --gauss-root "$GAUSS_ROOT" --index "$PREP_ROOT/nuscenes.index.json" \
  --output-dir "$PREP_ROOT/gaussian-inventory" --inventory-only
```

inventory.jsonl 记录真实文件路径、author_scene_key、author_frame_key、view 和大小。scene_0 到 scene_21 只是作者编号，不能对应 SDK 第0到21项；00000 也可能是视频帧、sweep 或重建内部索引。

可以自动识别文件名中保留的真实 sample token、keyframe sample_data token 或唯一原始相机完整文件名。只有数字编号时，需要作者导出索引、重建输入图像列表或带原图路径的 metadata 来建立映射。

请产出 verified_author_frames.jsonl，每行写明 author_scene_key、author_frame_key、sample_token、evidence，以及 coordinate_frame 或 gauss_to_ego。可选 view 字段用于每视角各自映射。evidence 应指向具体作者 metadata、输入图片或索引文件；不能填写“按排序猜测”。author_frame_key 保留文件原零填充，例如 00000。

坐标约定必须查清：

| 确认的原高斯坐标 | 对应处理 |
|---|---|
| 当前参考 ego | coordinate_frame=current_ego |
| 对应输入相机 OpenCV 坐标 | coordinate_frame=camera，从 SDK 标定转到当前 ego |
| nuScenes world/global 坐标 | coordinate_frame=world，从 SDK ego pose 转到当前 ego |
| 作者自定义重建坐标 | 提供真实 gauss_to_ego 4×4 刚体变换；归一化尺度还需单独恢复单位 |

没有坐标依据时保持未匹配。不能默认给 identity 矩阵，也不能用模型效果看起来合理来证明坐标正确。

将已确认的场景帧映射展开为每个文件的映射，再执行关联：

```bash
python -m gaussiandwm_research.expand_gaussian_mapping \
  --inventory "$PREP_ROOT/gaussian-inventory/inventory.jsonl" \
  --frame-map "$PREP_ROOT/verified_author_frames.jsonl" \
  --output "$PREP_ROOT/gaussian.files.map.jsonl" \
  --unmatched-output "$PREP_ROOT/gaussian.unmapped.jsonl"

python -m gaussiandwm_research.match_gaussians \
  --gauss-root "$GAUSS_ROOT" --index "$PREP_ROOT/nuscenes.index.json" \
  --mapping "$PREP_ROOT/gaussian.files.map.jsonl" \
  --output-dir "$PREP_ROOT/gaussian-matched"

python -m gaussiandwm_research.join_public_data --mode offline \
  --qa-jsonl "$PREP_ROOT/nuinteract/train.qa.jsonl" "$PREP_ROOT/nuinteract/val.qa.jsonl" \
  --index "$PREP_ROOT/nuscenes.index.json" \
  --gaussian-map "$PREP_ROOT/gaussian-matched/frames.gaussians.jsonl" \
  --output-dir "$PREP_ROOT/offline-matched" --check-images
```

默认需要同 sample_token 六个视角均有确认的 Gaussian 文件。部分视角实验可以显式用 --min-views，但基线与改进必须使用相同视角集。多个重建版本不能都放同目录让工具随便挑。

离线 QA 清单不依赖 LiDAR，制作当前帧标定与 CLIP 查询特征：

```bash
for SPLIT in train val; do
  python -m gaussiandwm_research.prepare_nuscenes \
    --nuscenes-root "$NUSC_ROOT" --version v1.0-trainval --split "$SPLIT" \
    --qa-jsonl "$PREP_ROOT/offline-matched/$SPLIT.matched.qa.jsonl" \
    --output "$PREP_ROOT/offline.$SPLIT.raw.jsonl" --horizon 0 --depth-mode none
  python -m gaussiandwm_research.prepare_semantics \
    --manifest "$PREP_ROOT/offline.$SPLIT.raw.jsonl" --data-root "$NUSC_ROOT" \
    --output-manifest "$PREP_ROOT/offline.$SPLIT.jsonl" \
    --cache-root "$PREP_ROOT/semantic" --text-only
done
```

输出保留 Gaussian transforms、QA 原标签、source、qa_group、split。后续 offline_similarity 和 offline_cascade 用同一份清单；没有实例 ids 时不能声称完成同实例语义填充。

## 6 数据准入与回传

这些检查由实验执行者在数据机器上运行，不是模型性能测试。离线：

```bash
python -m gaussiandwm_research.audit_public_manifest \
  --manifest "$PREP_ROOT/offline.train.jsonl" "$PREP_ROOT/offline.val.jsonl" \
  --index "$PREP_ROOT/nuscenes.index.json" --data-root "$NUSC_ROOT" \
  --mode offline --require-features --output-dir "$PREP_ROOT/offline-admission"
```

在线完整监督使用 online.train.jsonl/online.val.jsonl，设置 --mode online --require-features --require-world。检查失败时保留报告，修复具体来源或重新筛选完整样本，不能只忽略错误状态。

请回传以下轻量制品，不必上传大量图像、特征、Gaussian 或 LiDAR：

1. source-sha256.txt，以及 SDK version/metadata SHA256。
2. 每一步 report.json；异常较多时附 rejected.jsonl 前20条。
3. train/val 的 QA 数量、独立 sample 数量、scene token 列表及类别统计。
4. Gaussian 已匹配/未匹配文件数、确认映射证据、坐标约定和实际覆盖的场景范围。
5. 最终清单各2条脱离大文件的样例，用来确认字段与路径；注意保留真实身份，不自行造例。

数据准入至少满足：每个接受样本有唯一帧身份；六视角属于同一 sample；train/val 场景无交叉；3D embedding 占位符已接回真实标签；无缺失所需输入；离线 Gaussian 的帧身份和坐标系均有证据。原始范围外、未来帧不足和缺失传感器造成的筛选要报告数量。

全量 Gaussian 或作者映射仍缺失时，回传未匹配清单及缺少的 metadata 项，先完成不依赖这些 Gaussian 的在线数据路线。有公开真实子集可用时，应在同一子集比较方法，并注明覆盖范围；不把子集结果写成850场景完整复现。

## 给数据处理助手的任务文本

请先阅读本仓库 docs/PUBLIC_DATA_TASK.md 和 docs/DATA_CONTRACT.md，仅执行公开数据转换、身份匹配和数据准入。使用真实 NuInteract/OmniDrive 标签和 nuScenes SDK 索引，保留原始标签、坐标约定、来源划分和匹配证据。先跑 NuInteract 转换与在线 QA 关联，再清点作者 Gaussian；只有明确 sample token/原图/作者索引及坐标依据才生成离线映射。不得按 scene 数字、目录排序、最近帧或 identity 矩阵猜配；不得生成假 QA、用空数据补缺或把伪深度作为测量 GT。缺失项记录到 rejected.jsonl，回传上述轻量制品和实际覆盖量。不要修改模型架构，也不要把数据处理完成说成论文指标已复现。
