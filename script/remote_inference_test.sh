#!/usr/bin/bash

set -euo pipefail

source /inspire/hdd/global_user/liuhairuo-253208120281/miniconda3/bin/activate
conda activate lingbot-va

POLICY_HOST=${POLICY_HOST:-"127.0.0.1"}
POLICY_PORT=${POLICY_PORT:-"29536"}
DATASET_NAME=${DATASET_NAME:-"lift_box_high"}
VIDEO_CHUNK_DIR=${VIDEO_CHUNK_DIR:-"/inspire/hdd/global_user/liuhairuo-253208120281/cache/huggingface/lerobot/${DATASET_NAME}/videos/chunk-000"}
EPISODE_ID=${EPISODE_ID:-"000000"}
PROMPT=${PROMPT:-"Lift the box high."}
VIDEO_EVAL_MODE=${VIDEO_EVAL_MODE:-"robotwin"}
VIDEO_MAX_CHUNKS=${VIDEO_MAX_CHUNKS:-"1"}

echo "[config] DATASET_NAME=${DATASET_NAME}"
echo "[config] VIDEO_CHUNK_DIR=${VIDEO_CHUNK_DIR}"
echo "[config] EPISODE_ID=${EPISODE_ID}"
echo "[config] PROMPT=${PROMPT}"
echo "[config] VIDEO_EVAL_MODE=${VIDEO_EVAL_MODE}"
echo "[config] VIDEO_MAX_CHUNKS=${VIDEO_MAX_CHUNKS}"

python -m wan_va.utils.Simple_Remote_Infer.deploy.remote_inference_lerobot_norobot \
  --policy_host "${POLICY_HOST}" \
  --policy_port "${POLICY_PORT}" \
  --prompt "${PROMPT}" \
  --video_chunk_dir "${VIDEO_CHUNK_DIR}" \
  --episode_id "${EPISODE_ID}" \
  --video_eval_mode "${VIDEO_EVAL_MODE}" \
  --video_max_chunks "${VIDEO_MAX_CHUNKS}"
