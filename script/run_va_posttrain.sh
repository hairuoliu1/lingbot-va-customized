#!/usr/bin/bash

NGPU=${PET_NPROC_PER_NODE:-"8"}
NNODES=${PET_NNODES:-"1"}
NODE_RANK=${PET_NODE_RANK:-"0"}
MASTER_ADDR=${PET_MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${PET_MASTER_PORT:-"29501"}
TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}
CONFIG_NAME=${CONFIG_NAME:-"flexiv_train"}

export NCCL_DEBUG=WARN
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

readonly PROJ="/inspire/hdd/global_user/liuhairuo-253208120281"
export HF_LEROBOT_HOME="${PROJ}/cache/huggingface/lerobot"
export CACHE_ROOT="${PROJ}/cache"
source "${PROJ}/miniconda3/bin/activate"
conda activate lingbot-va
cd "${PROJ}/lingbot-va"

export TOKENIZERS_PARALLELISM=false
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE} \
python -m torch.distributed.run \
    --nnodes=${NNODES} \
    --nproc_per_node=${NGPU} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port ${MASTER_PORT} \
    -m wan_va.train --config-name ${CONFIG_NAME} "$@"
