#!/usr/bin/env bash

set -euo pipefail

source /inspire/hdd/global_user/liuhairuo-253208120281/miniconda3/bin/activate
conda activate lingbot-va

NGPU=${NGPU:-8}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
REPO_ROOT=${REPO_ROOT:-/inspire/hdd/global_user/liuhairuo-253208120281/cache/huggingface/lerobot}
MODEL_ROOT=${MODEL_ROOT:-/inspire/hdd/global_user/liuhairuo-253208120281/model/robbyant/lingbot-va-base}
TARGET_FPS=${TARGET_FPS:-10}
MAX_AREA=${MAX_AREA:-$((256*256))}
FRAME_COUNT_ALIGN=${FRAME_COUNT_ALIGN:-trim}
VIDEO_BACKEND=${VIDEO_BACKEND:-pyav}
BATCH_SIZE=${BATCH_SIZE:-16}
VIDEO_NUM_WORKERS=${VIDEO_NUM_WORKERS:-8}

export CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS

IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
if [[ ${#GPU_IDS[@]} -eq 0 ]]; then
  echo "[error] no GPU ids found in CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  exit 1
fi

if (( NGPU > ${#GPU_IDS[@]} )); then
  echo "[warn] NGPU=${NGPU} > visible GPUs=${#GPU_IDS[@]}, reduce NGPU."
  NGPU=${#GPU_IDS[@]}
fi

mapfile -t REPO_IDS < <(
  for p in "${REPO_ROOT}"/*; do
    [[ -e "${p}" ]] || continue
    # Accept real directories and symlinked dataset directories.
    if [[ -d "${p}" && -f "${p}/meta/info.json" ]]; then
      basename "${p}"
    fi
  done | sort
)

if [[ ${#REPO_IDS[@]} -eq 0 ]]; then
  echo "[error] no LeRobot repos found under ${REPO_ROOT}"
  exit 1
fi

echo "[info] found ${#REPO_IDS[@]} repos under ${REPO_ROOT}"
echo "[info] launch ${NGPU} workers on GPUs: ${GPU_IDS[*]}"

for ((worker=0; worker<NGPU; worker++)); do
  gpu="${GPU_IDS[$worker]}"
  (
    echo "[info] worker=${worker} gpu=${gpu} started"
    for ((i=worker; i<${#REPO_IDS[@]}; i+=NGPU)); do
      repo_id="${REPO_IDS[$i]}"
      echo "[info] worker=${worker} gpu=${gpu} repo=${repo_id}"
      CUDA_VISIBLE_DEVICES="${gpu}" \
      python script/extract_lerobot_latents.py \
        --repo-id "${repo_id}" \
        --repo-root "${REPO_ROOT}" \
        --target-fps "${TARGET_FPS}" \
        --max-area "${MAX_AREA}" \
        --model-root "${MODEL_ROOT}" \
        --frame-count-align "${FRAME_COUNT_ALIGN}" \
        --device cuda:0 \
        --batch-size "${BATCH_SIZE}" \
        --video-backend "${VIDEO_BACKEND}" \
        --video-num-workers "${VIDEO_NUM_WORKERS}"
    done
    echo "[info] worker=${worker} gpu=${gpu} finished"
  ) &
done

wait
echo "[info] all workers finished"
