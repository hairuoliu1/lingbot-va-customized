#!/usr/bin/bash

NGPU=${PET_NPROC_PER_NODE:-"8"}
NNODES=${PET_NNODES:-"1"}
NODE_RANK=${PET_NODE_RANK:-"0"}
MASTER_ADDR=${PET_MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${PET_MASTER_PORT:-"29501"}
TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}
CONFIG_NAME=${CONFIG_NAME:-"flexiv_train"}
EXP_NAME=${EXP_NAME:-""}
RESUME_FROM=${RESUME_FROM:-""}
DEFAULT_DATASET_PATH="/inspire/hdd/global_user/liuhairuo-253208120281/cache/huggingface/lerobot"
DATASET_PATH=${DATASET_PATH:-""}

# Support:
#   bash run_va_posttrain.sh /path/to/dataset
#   bash run_va_posttrain.sh /path/to/dataset my_exp_name
#   bash run_va_posttrain.sh /path/to/dataset my_exp_name /path/to/checkpoint_step_xxx
if [[ -z "${DATASET_PATH}" && $# -gt 0 ]]; then
    DATASET_PATH="$1"
    shift
fi
if [[ -z "${EXP_NAME}" && $# -gt 0 ]]; then
    EXP_NAME="$1"
    shift
fi
if [[ -z "${RESUME_FROM}" && $# -gt 0 ]]; then
    RESUME_FROM="$1"
    shift
fi
if [[ -z "${DATASET_PATH}" ]]; then
    DATASET_PATH="${DEFAULT_DATASET_PATH}"
fi

export NCCL_DEBUG=ERROR
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

readonly PROJ="/inspire/hdd/global_user/liuhairuo-253208120281"
export HF_LEROBOT_HOME="${PROJ}/cache/huggingface/lerobot"
export CACHE_ROOT="${PROJ}/cache"
source "${PROJ}/miniconda3/bin/activate"
conda activate lingbot-va
cd "${PROJ}/lingbot-va"

export TOKENIZERS_PARALLELISM=false
extra_args=()
if [[ -n "${DATASET_PATH}" ]]; then
    extra_args+=(--dataset-path "${DATASET_PATH}")
fi
if [[ -n "${EXP_NAME}" ]]; then
    extra_args+=(--exp-name "${EXP_NAME}")
fi
if [[ -n "${RESUME_FROM}" ]]; then
    extra_args+=(--resume-from "${RESUME_FROM}")
fi
echo "[run_va_posttrain] CONFIG_NAME=${CONFIG_NAME}"
echo "[run_va_posttrain] DATASET_PATH=${DATASET_PATH}"
if [[ -n "${EXP_NAME}" ]]; then
    echo "[run_va_posttrain] EXP_NAME=${EXP_NAME}"
fi
if [[ -n "${RESUME_FROM}" ]]; then
    echo "[run_va_posttrain] RESUME_FROM=${RESUME_FROM}"
fi
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE} \
python -m torch.distributed.run \
    --nnodes=${NNODES} \
    --nproc_per_node=${NGPU} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port ${MASTER_PORT} \
    -m wan_va.train --config-name ${CONFIG_NAME} "${extra_args[@]}" "$@"
