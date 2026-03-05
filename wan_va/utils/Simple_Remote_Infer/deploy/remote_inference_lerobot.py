#!/usr/bin/env python
"""Real-robot Flexiv remote inference client for LingBot-VA websocket server."""

import argparse
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pyrealsense2 as rs
import spdlog

import flexivrdk

try:
    from .websocket_client_policy import WebsocketClientPolicy
except ImportError:
    from websocket_client_policy import WebsocketClientPolicy


class RealSenseCameraManager:
    def __init__(
        self,
        serial_numbers: List[str],
        width: int,
        height: int,
        fps: int,
        logger: spdlog.Logger,
    ):
        self.serial_numbers = serial_numbers
        self.width = width
        self.height = height
        self.fps = fps
        self.logger = logger
        self.pipelines: Dict[str, rs.pipeline] = {}

    def start(self):
        ctx = rs.context()
        devices = ctx.query_devices()
        available = [d.get_info(rs.camera_info.serial_number) for d in devices]
        self.logger.info(f"Available RealSense devices: {available}")

        for sn in self.serial_numbers:
            if sn not in available:
                raise RuntimeError(f"RealSense {sn} not found in connected devices.")

            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(sn)
            config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
            pipeline.start(config)
            for _ in range(5):
                pipeline.wait_for_frames(timeout_ms=1000)
            self.pipelines[sn] = pipeline
            self.logger.info(f"RealSense {sn} started")

    def get_frames(self) -> Dict[str, np.ndarray]:
        frames: Dict[str, np.ndarray] = {}
        for sn, pipeline in self.pipelines.items():
            frameset = pipeline.wait_for_frames(timeout_ms=1000)
            color = frameset.get_color_frame()
            if not color:
                raise RuntimeError(f"No color frame from camera {sn}")
            frame = np.asanyarray(color.get_data())
            if frame.shape[:2] != (self.height, self.width):
                frame = cv2.resize(frame, (self.width, self.height))
            frames[sn] = frame
        return frames

    def stop(self):
        for sn, pipeline in self.pipelines.items():
            try:
                pipeline.stop()
                self.logger.info(f"RealSense {sn} stopped")
            except Exception as exc:
                self.logger.warn(f"Stop camera {sn} failed: {exc}")
        self.pipelines.clear()


def init_robot(
    logger: spdlog.Logger,
    robot_sn: str,
    gripper_name: str,
    home_pos: List[float],
    home_rpy: List[float],
) -> Tuple[flexivrdk.Robot, flexivrdk.Gripper]:
    logger.info(f"[{robot_sn}] initializing")
    robot = flexivrdk.Robot(robot_sn)

    if robot.fault():
        logger.warn(f"[{robot_sn}] clearing fault")
        robot.ClearFault()
        time.sleep(0.1)
        if robot.fault():
            raise RuntimeError(f"[{robot_sn}] fault cannot be cleared")

    robot.Enable()
    while not robot.operational():
        time.sleep(0.1)

    gripper = flexivrdk.Gripper(robot)
    gripper.Enable(gripper_name)
    gripper.Move(0.085, 0.1, 50)
    while robot.busy():
        time.sleep(0.1)

    robot.SwitchMode(flexivrdk.Mode.NRT_PRIMITIVE_EXECUTION)
    home_coord = flexivrdk.Coord(home_pos, home_rpy, ["WORLD", "WORLD_ORIGIN"])
    robot.ExecutePrimitive("MoveL", {"target": home_coord, "vel": 0.2})
    while not robot.primitive_states()["reachedTarget"]:
        time.sleep(0.02)

    robot.ExecutePrimitive("ZeroFTSensor", dict())
    while not robot.primitive_states()["terminated"]:
        time.sleep(0.1)

    robot.SwitchMode(flexivrdk.Mode.NRT_CARTESIAN_MOTION_FORCE)
    logger.info(f"[{robot_sn}] ready")
    return robot, gripper


def get_robot_state(
    left_robot: flexivrdk.Robot,
    right_robot: flexivrdk.Robot,
    left_gripper: flexivrdk.Gripper,
    right_gripper: flexivrdk.Gripper,
) -> np.ndarray:
    left_tcp = left_robot.states().tcp_pose
    right_tcp = right_robot.states().tcp_pose
    left_grip = left_gripper.states().width
    right_grip = right_gripper.states().width
    return np.array([*left_tcp[:7], left_grip, *right_tcp[:7], right_grip], dtype=np.float32)


def execute_action(
    action: np.ndarray,
    left_robot: flexivrdk.Robot,
    right_robot: flexivrdk.Robot,
    left_gripper: flexivrdk.Gripper,
    right_gripper: flexivrdk.Gripper,
):
    left_pose = action[0:7].tolist()
    right_pose = action[8:15].tolist()
    left_grip = float(np.clip(abs(action[7]), 0.0001, 0.085))
    right_grip = float(np.clip(abs(action[15]), 0.0001, 0.085))

    left_robot.SendCartesianMotionForce(left_pose, [0.0] * 6)
    right_robot.SendCartesianMotionForce(right_pose, [0.0] * 6)
    left_gripper.Move(left_grip, 0.1, 50)
    right_gripper.Move(right_grip, 0.1, 50)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--left_robot_sn", required=True)
    parser.add_argument("--right_robot_sn", required=True)
    parser.add_argument("--left_gripper_name", default="Robotiq-2F-85")
    parser.add_argument("--right_gripper_name", default="Robotiq-2F-85")

    parser.add_argument("--policy_camera1_sn", required=True)
    parser.add_argument("--policy_camera2_sn", required=True)
    parser.add_argument("--policy_camera3_sn", required=True)
    parser.add_argument("--policy_width", type=int, default=640)
    parser.add_argument("--policy_height", type=int, default=480)
    parser.add_argument("--policy_fps", type=int, default=30)

    parser.add_argument("--policy_host", default="127.0.0.1")
    parser.add_argument("--policy_port", type=int, default=1106)
    parser.add_argument("--frequency", type=int, default=30)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max_steps", type=int, default=0, help="0 means run forever")
    parser.add_argument("--action_dim", type=int, default=16)
    parser.add_argument("--kv_every_action_steps", type=int, default=12)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--dry_run", action="store_true", help="Do not send action to robots")

    parser.add_argument("--left_home_pos", nargs=3, type=float, default=[0.50, 0.20, 0.25])
    parser.add_argument("--right_home_pos", nargs=3, type=float, default=[0.50, -0.20, 0.25])
    parser.add_argument("--home_rpy", nargs=3, type=float, default=[0.0, 160.0, 0.0])
    return parser.parse_args()


def main():
    args = parse_args()
    logger = spdlog.ConsoleLogger("lingbot_realrobot")

    camera_sns = [args.policy_camera1_sn, args.policy_camera2_sn, args.policy_camera3_sn]
    camera_keys = ["observation.images.cam1", "observation.images.cam2", "observation.images.cam3"]
    sn_to_key = dict(zip(camera_sns, camera_keys))

    camera_manager = RealSenseCameraManager(
        serial_numbers=camera_sns,
        width=args.policy_width,
        height=args.policy_height,
        fps=args.policy_fps,
        logger=logger,
    )
    camera_manager.start()

    left_robot, left_gripper = init_robot(
        logger,
        args.left_robot_sn,
        args.left_gripper_name,
        args.left_home_pos,
        args.home_rpy,
    )
    right_robot, right_gripper = init_robot(
        logger,
        args.right_robot_sn,
        args.right_gripper_name,
        args.right_home_pos,
        args.home_rpy,
    )

    client = WebsocketClientPolicy(host=args.policy_host, port=args.policy_port)
    client.infer({"reset": True, "prompt": args.prompt})
    logger.info("LingBot-VA reset done")

    control_period = 1.0 / args.frequency
    frame_idx = 0
    kv_every_action_steps = max(1, int(args.kv_every_action_steps))

    pending_actions: List[np.ndarray] = []
    pending_chunk_action: Optional[np.ndarray] = None
    pending_key_frames: List[Dict[str, np.ndarray]] = []
    first_chunk = True
    executed_action_steps = 0

    try:
        while True:
            loop_start = time.perf_counter()

            raw_frames = camera_manager.get_frames()  # BGR
            images_rgb: Dict[str, np.ndarray] = {}
            for sn, bgr in raw_frames.items():
                key = sn_to_key.get(sn)
                if key is None:
                    continue
                images_rgb[key] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            if len(images_rgb) != 3:
                raise RuntimeError(f"Expect 3 camera frames, got {list(images_rgb.keys())}")

            if not pending_actions:
                infer_ret = client.infer({"obs": images_rgb, "prompt": args.prompt})
                if "action" not in infer_ret:
                    raise RuntimeError(f"LingBot response missing action: {infer_ret}")

                pending_chunk_action = np.asarray(infer_ret["action"], dtype=np.float32)
                if pending_chunk_action.ndim != 3 or pending_chunk_action.shape[0] != args.action_dim:
                    raise RuntimeError(
                        f"Unexpected action chunk shape {pending_chunk_action.shape}, "
                        f"expected [{args.action_dim}, F, H]."
                    )

                f_dim, h_dim = int(pending_chunk_action.shape[1]), int(pending_chunk_action.shape[2])
                start_i = 1 if first_chunk else 0
                if start_i >= f_dim:
                    start_i = 0

                pending_actions = [
                    pending_chunk_action[:, i, j].astype(np.float32)
                    for i in range(start_i, f_dim)
                    for j in range(h_dim)
                ]
                pending_key_frames = []

                if args.debug:
                    logger.info(
                        f"[chunk] shape={pending_chunk_action.shape}, "
                        f"kv_every_action_steps={kv_every_action_steps}, start_i={start_i}"
                    )

            action = pending_actions.pop(0)
            executed_action_steps += 1

            if args.debug and frame_idx % 10 == 0:
                state = get_robot_state(left_robot, right_robot, left_gripper, right_gripper)
                logger.info(
                    f"[step {frame_idx}] state_norm={np.linalg.norm(state):.4f}, "
                    f"action_norm={np.linalg.norm(action):.4f}"
                )

            if not args.dry_run:
                execute_action(
                    action,
                    left_robot,
                    right_robot,
                    left_gripper,
                    right_gripper,
                )

            if pending_chunk_action is not None and executed_action_steps % kv_every_action_steps == 0:
                pending_key_frames.append(images_rgb)

            if not pending_actions and pending_chunk_action is not None:
                if not pending_key_frames:
                    pending_key_frames = [images_rgb]

                client.infer(
                    {
                        "obs": pending_key_frames,
                        "compute_kv_cache": True,
                        "state": pending_chunk_action.astype(np.float32),
                    }
                )
                pending_chunk_action = None
                first_chunk = False

            frame_idx += 1
            if args.max_steps > 0 and frame_idx >= args.max_steps:
                break

            elapsed = time.perf_counter() - loop_start
            if elapsed < control_period:
                time.sleep(control_period - elapsed)
            elif args.debug:
                logger.warn(f"[timing] loop={elapsed * 1000:.1f}ms > {control_period * 1000:.1f}ms")

    except KeyboardInterrupt:
        logger.info("Interrupted")
    finally:
        camera_manager.stop()
        logger.info(f"Done. steps={frame_idx}")


if __name__ == "__main__":
    main()
