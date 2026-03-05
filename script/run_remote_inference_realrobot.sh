#!/usr/bin/bash

set -euo pipefail

# Optional: CONDA_ENV=xxx ./script/run_remote_inference_realrobot.sh
if [ -n "${CONDA_ENV:-}" ]; then
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV}"
fi

LEFT_ROBOT_SN=${LEFT_ROBOT_SN:-"Rizon 4s-063005"}
RIGHT_ROBOT_SN=${RIGHT_ROBOT_SN:-"Rizon 4s-062987"}

LEFT_GRIPPER_NAME=${LEFT_GRIPPER_NAME:-"Robotiq-2F-85"}
RIGHT_GRIPPER_NAME=${RIGHT_GRIPPER_NAME:-"Robotiq-2F-85"}

POLICY_CAMERA1_SN=${POLICY_CAMERA1_SN:-"<CAM1_SN>"}
POLICY_CAMERA2_SN=${POLICY_CAMERA2_SN:-"<CAM2_SN>"}
POLICY_CAMERA3_SN=${POLICY_CAMERA3_SN:-"<CAM3_SN>"}

POLICY_WIDTH=${POLICY_WIDTH:-"640"}
POLICY_HEIGHT=${POLICY_HEIGHT:-"480"}
POLICY_FPS=${POLICY_FPS:-"30"}

POLICY_HOST=${POLICY_HOST:-"127.0.0.1"}
POLICY_PORT=${POLICY_PORT:-"1106"}
FREQUENCY=${FREQUENCY:-"30"}
PROMPT=${PROMPT:-"Put the chain into the box."}
MAX_STEPS=${MAX_STEPS:-"0"}
KV_EVERY_ACTION_STEPS=${KV_EVERY_ACTION_STEPS:-"12"}

if [[ "${POLICY_CAMERA1_SN}" == "<CAM1_SN>" || "${POLICY_CAMERA2_SN}" == "<CAM2_SN>" || "${POLICY_CAMERA3_SN}" == "<CAM3_SN>" ]]; then
  echo "Please set POLICY_CAMERA1_SN / POLICY_CAMERA2_SN / POLICY_CAMERA3_SN before running."
  exit 1
fi

python -m wan_va.utils.Simple_Remote_Infer.deploy.remote_inference_lerobot \
  --left_robot_sn "${LEFT_ROBOT_SN}" \
  --right_robot_sn "${RIGHT_ROBOT_SN}" \
  --left_gripper_name "${LEFT_GRIPPER_NAME}" \
  --right_gripper_name "${RIGHT_GRIPPER_NAME}" \
  --policy_camera1_sn "${POLICY_CAMERA1_SN}" \
  --policy_camera2_sn "${POLICY_CAMERA2_SN}" \
  --policy_camera3_sn "${POLICY_CAMERA3_SN}" \
  --policy_width "${POLICY_WIDTH}" \
  --policy_height "${POLICY_HEIGHT}" \
  --policy_fps "${POLICY_FPS}" \
  --policy_host "${POLICY_HOST}" \
  --policy_port "${POLICY_PORT}" \
  --frequency "${FREQUENCY}" \
  --prompt "${PROMPT}" \
  --max_steps "${MAX_STEPS}" \
  --kv_every_action_steps "${KV_EVERY_ACTION_STEPS}" \
  "$@"
