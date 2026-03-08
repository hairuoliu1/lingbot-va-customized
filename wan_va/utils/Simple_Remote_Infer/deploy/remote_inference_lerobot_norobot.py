#!/usr/bin/env python
"""No-robot, dummy-observation test runner for LingBot-VA remote inference."""

import argparse
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def build_lingbot_client(host: str, port: int):
    try:
        from wan_va.utils.Simple_Remote_Infer.deploy.websocket_client_policy import WebsocketClientPolicy
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Cannot import wan_va. Please run `pip install -e /path/to/lingbot-va` first."
        ) from exc
    return WebsocketClientPolicy(host=host, port=port)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_host", default="127.0.0.1")
    parser.add_argument("--policy_port", type=int, default=1106)
    parser.add_argument("--frequency", type=int, default=30)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max_steps", type=int, default=200, help="0 means run forever")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--kv_every_action_steps",
        type=int,
        default=15,
        help="Sample one key frame every N executed action steps for compute_kv_cache.",
    )
    parser.add_argument(
        "--video_chunk_dir",
        type=str,
        default="",
        help=(
            "If set, load videos from observation.images.cam1/cam2/cam3 under this chunk dir "
            "and run video-based inference."
        ),
    )
    parser.add_argument(
        "--episode_id",
        type=str,
        default="000000",
        help="Preferred episode id when --video_chunk_dir is set, e.g. 000000.",
    )
    parser.add_argument(
        "--video_eval_mode",
        type=str,
        default="robotwin",
        choices=["robotwin", "oneshot"],
        help="Video inference mode. 'robotwin' aligns with evaluation/robotwin flow.",
    )
    parser.add_argument(
        "--video_max_chunks",
        type=int,
        default=1,
        help="How many infer->compute_kv_cache chunks to run in video mode.",
    )
    return parser.parse_args()


def make_dummy_obs(rng: np.random.Generator, height: int, width: int) -> Dict[str, np.ndarray]:
    return {
        "observation.images.cam1": rng.integers(0, 256, (height, width, 3), dtype=np.uint8),
        "observation.images.cam2": rng.integers(0, 256, (height, width, 3), dtype=np.uint8),
        "observation.images.cam3": rng.integers(0, 256, (height, width, 3), dtype=np.uint8),
    }


def _read_frame_rgb(video_path: Path, frame_idx: int = 0) -> np.ndarray:
    # For mp4 datasets (often AV1), prefer ffmpeg software decode to avoid noisy OpenCV failures.
    if video_path.suffix.lower() not in {".mp4", ".mkv", ".webm"}:
        cap = cv2.VideoCapture(str(video_path))
        if frame_idx > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame_bgr = cap.read()
        cap.release()
        if ok and frame_bgr is not None:
            return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.uint8)

    # Fallback: use ffmpeg software decoding (works for AV1 on machines where OpenCV fails).
    vf_select = f"select=eq(n\\,{int(frame_idx)})"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        vf_select,
        "-frames:v",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "png",
        "-",
    ]
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Failed to read first frame from {video_path}. "
            "OpenCV decode failed and `ffmpeg` is not installed."
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(
            f"Failed to read first frame from {video_path}. ffmpeg decode error: {stderr}"
        ) from exc

    buf = np.frombuffer(proc.stdout, dtype=np.uint8)
    frame_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        raise RuntimeError(
            f"Failed to read frame {frame_idx} from {video_path}. ffmpeg produced unreadable frame bytes."
        )
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.uint8)


def resolve_video_paths(video_chunk_dir: Path, episode_id: str) -> Dict[str, Path]:
    cam_to_obs_key = {
        "cam1": "observation.images.cam1",
        "cam2": "observation.images.cam2",
        "cam3": "observation.images.cam3",
    }
    video_paths: Dict[str, Path] = {}

    for cam_name, obs_key in cam_to_obs_key.items():
        cam_dir = video_chunk_dir / f"observation.images.{cam_name}"
        preferred = cam_dir / f"episode_{episode_id}.mp4"
        if preferred.exists():
            video_path = preferred
        else:
            candidates = sorted(cam_dir.glob("episode_*.mp4"))
            if not candidates:
                raise FileNotFoundError(f"No videos found under {cam_dir}")
            video_path = candidates[0]
        video_paths[obs_key] = video_path
    return video_paths


def load_obs_at_frame(video_paths: Dict[str, Path], frame_idx: int) -> Dict[str, np.ndarray]:
    obs: Dict[str, np.ndarray] = {}
    for obs_key, video_path in video_paths.items():
        obs[obs_key] = _read_frame_rgb(video_path, frame_idx=frame_idx)
        print(f"[loaded] {obs_key} <- {video_path.name}, frame={frame_idx}, shape={obs[obs_key].shape}")
    return obs


def print_action_2d(action: np.ndarray, *, skip_first_frame: bool = False) -> None:
    if skip_first_frame and action.ndim == 3 and action.shape[1] > 1:
        action = action[:, 1:, :]
        print("[skip] dropped first frame actions for display.")
    print(f"[inference ok] action shape={action.shape}, dtype={action.dtype}")
    action_2d = action.transpose(1, 2, 0).reshape(-1, action.shape[0])  # [C,F,N] -> [F*N,C]
    print(f"[reshape] action_2d shape={action_2d.shape}")
    print(action_2d)


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    client = build_lingbot_client(host=args.policy_host, port=args.policy_port)
    client.infer({"reset": True, "prompt": args.prompt})
    print("LingBot-VA reset done.")

    if args.video_chunk_dir:
        video_paths = resolve_video_paths(Path(args.video_chunk_dir), args.episode_id)
        frame_idx = 0
        first_obs = load_obs_at_frame(video_paths, frame_idx=frame_idx)

        if args.video_eval_mode == "oneshot":
            infer_ret = client.infer({"obs": first_obs, "prompt": args.prompt})
            if "action" not in infer_ret:
                raise RuntimeError(f"LingBot response missing action: {infer_ret}")
            action = np.asarray(infer_ret["action"], dtype=np.float32)
            print_action_2d(action, skip_first_frame=first)
            return

        # Align with evaluation/robotwin: infer one chunk, execute action steps,
        # collect key frames, then call compute_kv_cache with the executed chunk action.
        first = True
        obs_for_infer = first_obs
        max_chunks = max(1, int(args.video_max_chunks))
        for chunk_id in range(max_chunks):
            infer_ret = client.infer({"obs": obs_for_infer, "prompt": args.prompt})
            if "action" not in infer_ret:
                raise RuntimeError(f"LingBot response missing action: {infer_ret}")
            action = np.asarray(infer_ret["action"], dtype=np.float32)
            if action.ndim != 3:
                raise RuntimeError(f"Unexpected action shape {action.shape}, expected [C,F,N].")

            print(f"[chunk {chunk_id}]")
            print_action_2d(action)

            c_dim, f_dim, n_dim = int(action.shape[0]), int(action.shape[1]), int(action.shape[2])
            if c_dim != 16:
                print(f"[warn] action channel dim is {c_dim} (expected 16 for flexiv).")
            if n_dim % 4 == 0:
                action_per_frame = n_dim // 4
            else:
                action_per_frame = n_dim
                print(f"[warn] n_dim={n_dim} not divisible by 4, fallback action_per_frame={action_per_frame}.")

            key_frame_list: List[Dict[str, np.ndarray]] = []
            start_i = 1 if first else 0
            if start_i >= f_dim:
                start_i = 0

            for i in range(start_i, f_dim):
                for j in range(n_dim):
                    frame_idx += 1
                    if (j + 1) % action_per_frame == 0:
                        key_frame_list.append(load_obs_at_frame(video_paths, frame_idx=frame_idx))

            if not key_frame_list:
                key_frame_list = [load_obs_at_frame(video_paths, frame_idx=frame_idx)]

            client.infer(
                {
                    "obs": key_frame_list,
                    "compute_kv_cache": True,
                    "state": action.astype(np.float32),
                }
            )
            print(f"[kv] chunk={chunk_id}, key_frames={len(key_frame_list)}, frame_idx={frame_idx}")

            obs_for_infer = first_obs
            first = False
        return

    control_period = 1.0 / args.frequency
    frame_idx = 0

    pending_actions: List[np.ndarray] = []
    pending_chunk_action: Optional[np.ndarray] = None
    pending_key_frames: List[Dict[str, np.ndarray]] = []
    pending_chunk_consumed_steps = 0
    pending_step_indices: List[tuple[int, int]] = []
    first_chunk = True
    executed_action_steps = 0
    kv_every_action_steps = max(1, int(args.kv_every_action_steps))

    fake_state = np.zeros((16,), dtype=np.float32)

    try:
        while True:
            loop_start = time.perf_counter()
            images_rgb = make_dummy_obs(rng, args.height, args.width)

            if not pending_actions:
                infer_ret = client.infer({"obs": images_rgb, "prompt": args.prompt})
                if "action" not in infer_ret:
                    raise RuntimeError(f"LingBot response missing action: {infer_ret}")

                pending_chunk_action = np.asarray(infer_ret["action"], dtype=np.float32)
                if pending_chunk_action.ndim != 3 or pending_chunk_action.shape[0] != 16:
                    raise RuntimeError(
                        f"Unexpected action chunk shape {pending_chunk_action.shape}, expected [16, F, H]."
                    )

                f_dim, h_dim = int(pending_chunk_action.shape[1]), int(pending_chunk_action.shape[2])
                pending_chunk_consumed_steps = 0
                start_i = 1 if first_chunk else 0
                if start_i >= f_dim:
                    start_i = 0
                if args.debug:
                    print(
                        f"[chunk] F={f_dim}, H={h_dim}, kv_every_action_steps={kv_every_action_steps}, start_i={start_i}"
                    )
                pending_key_frames = []
                pending_step_indices = []
                pending_actions = []
                for i in range(start_i, f_dim):
                    for j in range(h_dim):
                        pending_actions.append(pending_chunk_action[:, i, j].astype(np.float32))
                        pending_step_indices.append((i, j))

            action = pending_actions.pop(0)
            _i_idx, _j_idx = pending_step_indices.pop(0)
            pending_chunk_consumed_steps += 1
            executed_action_steps += 1
            fake_state = action

            action_str = np.array2string(action, precision=4, suppress_small=True)
            print(f"[step {frame_idx}] action={action_str}")

            if pending_chunk_action is not None and executed_action_steps % kv_every_action_steps == 0:
                pending_key_frames.append(images_rgb)

            if not pending_actions and pending_chunk_action is not None:
                if not pending_key_frames:
                    pending_key_frames = [images_rgb]
                if args.debug:
                    print(f"[kv] key_frames={len(pending_key_frames)}")
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

    except KeyboardInterrupt:
        pass
    finally:
        print(f"Done. steps={frame_idx}, fake_state_norm={np.linalg.norm(fake_state):.4f}")


if __name__ == "__main__":
    main()
