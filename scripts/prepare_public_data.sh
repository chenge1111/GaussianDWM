#!/usr/bin/env bash
# Shared public QA + online calibrated-manifest preparation.
# Offline Gaussian matching is explicit; follow docs/PUBLIC_DATA_TASK.md section 5.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${NUSC_ROOT:?Set NUSC_ROOT to nuScenes root}"
: "${PUBLIC_ROOT:?Set PUBLIC_ROOT to downloaded NuInteract.zip directory}"
: "${PREP_ROOT:?Set PREP_ROOT to output directory}"
VERSION="${VERSION:-v1.0-trainval}"
DEPTH_MODE="${DEPTH_MODE:-lidar}"
HORIZON="${HORIZON:-6}"
FEATURE_MODE="${FEATURE_MODE:-full}"
mkdir -p "$PREP_ROOT"
python -m gaussiandwm_research.index_nuscenes \
  --nuscenes-root "$NUSC_ROOT" --version "$VERSION" --output "$PREP_ROOT/nuscenes.index.json"
sources=("$PUBLIC_ROOT/NuInteract.zip")
if [[ -f "$PUBLIC_ROOT/cap_public.tar.gz" ]]; then sources+=("$PUBLIC_ROOT/cap_public.tar.gz"); fi
python -m gaussiandwm_research.convert_public_qa --source nuinteract \
  --annotations "${sources[@]}" --index "$PREP_ROOT/nuscenes.index.json" --output-dir "$PREP_ROOT/nuinteract"
qa=("$PREP_ROOT/nuinteract/train.qa.jsonl" "$PREP_ROOT/nuinteract/val.qa.jsonl")
if [[ -n "${OMNI_ROOT:-}" ]]; then
  python -m gaussiandwm_research.convert_public_qa --source omnidrive \
    --annotations "$OMNI_ROOT/desc" "$OMNI_ROOT/vqa" "$OMNI_ROOT/conv" \
    --index "$PREP_ROOT/nuscenes.index.json" --output-dir "$PREP_ROOT/omnidrive"
  qa+=("$PREP_ROOT/omnidrive/train.qa.jsonl" "$PREP_ROOT/omnidrive/val.qa.jsonl")
fi
python -m gaussiandwm_research.join_public_data --mode online \
  --qa-jsonl "${qa[@]}" --index "$PREP_ROOT/nuscenes.index.json" \
  --output-dir "$PREP_ROOT/online-matched" --check-images
for SPLIT in train val; do
  SDK_SPLIT="$SPLIT"
  if [[ "$VERSION" == v1.0-mini ]]; then SDK_SPLIT="mini_$SPLIT"; fi
  python -m gaussiandwm_research.prepare_nuscenes \
    --nuscenes-root "$NUSC_ROOT" --version "$VERSION" --split "$SDK_SPLIT" \
    --qa-jsonl "$PREP_ROOT/online-matched/$SPLIT.matched.qa.jsonl" \
    --output "$PREP_ROOT/online.$SPLIT.raw.jsonl" \
    --horizon "$HORIZON" --depth-mode "$DEPTH_MODE" --depth-cache "$PREP_ROOT/depth"
  if [[ "$FEATURE_MODE" != none ]]; then
    feature_args=()
    if [[ "$FEATURE_MODE" == text ]]; then feature_args+=(--text-only); fi
    python -m gaussiandwm_research.prepare_semantics \
      --manifest "$PREP_ROOT/online.$SPLIT.raw.jsonl" --data-root "$NUSC_ROOT" \
      --output-manifest "$PREP_ROOT/online.$SPLIT.jsonl" --cache-root "$PREP_ROOT/semantic" \
      "${feature_args[@]}"
  fi
done
echo "Public data conversion completed. Inspect report.json/rejected.jsonl before admission."
