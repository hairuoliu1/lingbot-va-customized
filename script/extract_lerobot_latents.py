#!/usr/bin/env python3
"""
Extract Wan2.2 VAE latents for LeRobot-format datasets.

This follows LingBot-VA README requirements:
1) Read videos from <dataset>/videos/**/episode_<idx>.mp4
2) Respect action segments defined in meta/episodes.jsonl -> action_config
3) Resize frames to a VAE-friendly resolution (multiple of 32) near 256x256
4) Downsample to target fps
5) Encode with Wan2.2 VAE and (optionally) Wan text encoder
6) Save latents under <dataset>/latents/chunk-... using naming
   episode_{episode_index}_{start_frame}_{end_frame}.pth
"""

import argparse
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torchvision.transforms.functional as TF
from diffusers import AutoencoderKLWan
from tqdm import tqdm
from transformers import T5TokenizerFast, UMT5EncoderModel

from wan.utils.utils import best_output_size

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

try:
    import decord
except ImportError:
    decord = None

try:
    import av
except ImportError:
    av = None

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from torchcodec.decoders import VideoDecoder
except ImportError:
    VideoDecoder = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Add Wan2.2 VAE latents to a LeRobot dataset."
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="HuggingFace 数据集 repo_id（如 robbyant/robotwin-clean-and-aug-lerobot）。提供后会用 LeRobot 元数据解析视频路径。",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help=(
            "可选。批量模式下扫描该目录下所有 LeRobot 数据集子目录。"
            "例如 /.../cache/huggingface/lerobot"
        ),
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=10.0,
        help="FPS to sample before encoding.",
    )
    parser.add_argument(
        "--max-area",
        type=int,
        default=256 * 256,
        help="Target pixel area before encoding; output H/W stay divisible by 32.",
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=Path("/inspire/hdd/global_user/liuhairuo-253208120281/model/robbyant/lingbot-va-base"),
        help=(
            "LingBot-VA model root with subfolders: vae/, text_encoder/, tokenizer/."
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for VAE/text encoder, e.g., cuda or cpu.",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float32", "bfloat16"],
        help="Computation dtype for VAE.",
    )
    parser.add_argument(
        "--skip-text",
        action="store_true",
        help="Skip text embedding if you only need latents.",
    )
    parser.add_argument(
        "--frame-count-align",
        default="pad",
        choices=["pad", "trim"],
        help=(
            "Make sampled frame count satisfy (num_frames - 1) %% 4 == 0. "
            "'pad' repeats last frame; 'trim' drops tail frames."
        ),
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optional limit for debugging.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help=(
            "VAE encode batch size. Segments in a batch are temporally padded "
            "to the same frame count."
        ),
    )
    parser.add_argument(
        "--video-backend",
        default="pyav",
        choices=["pyav", "opencv", "decord", "torchcodec"],
        help=(
            "Video reader backend. Use pyav/opencv if decord fails on AV1 videos; "
            "torchcodec is also supported when installed."
        ),
    )
    parser.add_argument(
        "--video-num-workers",
        type=int,
        default=1,
        help="Number of decode threads used by the video backend when supported.",
    )
    args = parser.parse_args()
    if args.repo_id is None and args.repo_root is None:
        parser.error("At least one of --repo-id or --repo-root must be provided.")
    return args


def sample_frame_ids(
    start: int,
    end: int,
    ori_fps: float,
    target_fps: float,
    total_frames: int,
) -> List[int]:
    start = max(0, int(start))
    end = min(int(end), total_frames)
    if end <= start:
        return []
    duration = end - start
    target_count = max(1, int(round(duration * target_fps / max(ori_fps, 1e-6))))
    if target_count == 1:
        return [start]
    step = (end - start - 1) / float(target_count - 1)
    ids = [
        min(total_frames - 1, int(round(start + i * step)))
        for i in range(target_count)
    ]
    # keep stable ordering and remove accidental duplicates
    deduped = []
    for idx in ids:
        if not deduped or idx != deduped[-1]:
            deduped.append(idx)
    return deduped


def align_frame_ids_for_vae(frame_ids: List[int], mode: str) -> List[int]:
    """
    Wan2.2 temporal stride is 4, so valid input count is 1,5,9,...
    """
    if not frame_ids:
        return frame_ids

    remainder = (len(frame_ids) - 1) % 4
    if remainder == 0:
        return frame_ids

    if mode == "trim":
        keep = len(frame_ids) - remainder
        return frame_ids[:max(1, keep)]

    pad = 4 - remainder
    return frame_ids + [frame_ids[-1]] * pad


def preprocess_frames(
    frames: torch.Tensor, max_area: int
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    frames: (T, H, W, C) uint8
    Returns: video tensor (C, T, H, W) float in [-1,1] and (oh, ow)
    """
    if frames.numel() == 0:
        raise ValueError("Empty frame tensor.")
    t, h, w, _ = frames.shape
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid frame size: H={h}, W={w}")

    # Wan2.2 uses patch_size (1,2,2) with vae_stride (4,16,16) -> multiples of 32.
    dw = 2 * 16
    dh = 2 * 16
    safe_max_area = max(int(max_area), dw * dh)
    ow, oh = best_output_size(w, h, dw, dh, safe_max_area)
    if ow <= 0 or oh <= 0:
        raise ValueError(
            f"Invalid output size from best_output_size: {(oh, ow)} "
            f"(input={(h, w)}, max_area={safe_max_area})"
        )

    scale = max(ow / w, oh / h)
    resized_w = max(ow, int(round(w * scale)))
    resized_h = max(oh, int(round(h * scale)))

    frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W)
    frames = TF.resize(frames, [resized_h, resized_w], antialias=True)

    # center crop
    y0 = (resized_h - oh) // 2
    x0 = (resized_w - ow) // 2
    frames = frames[:, :, y0:y0 + oh, x0:x0 + ow]

    frames = frames.float().div_(255.0).sub_(0.5).div_(0.5)
    frames = frames.permute(1, 0, 2, 3).contiguous()  # (C, T, H, W)
    if oh % 32 != 0 or ow % 32 != 0:
        raise ValueError(f"Output size must be divisible by 32, got {(oh, ow)}")
    return frames, (oh, ow)


def init_vae(args, device, vae_dtype):
    vae_dir = args.model_root / "vae"
    if not vae_dir.exists():
        raise FileNotFoundError(f"VAE dir not found: {vae_dir}.")
    vae = AutoencoderKLWan.from_pretrained(str(vae_dir), torch_dtype=vae_dtype).to(device)
    vae.eval().requires_grad_(False)
    return {"mode": "lingbot_pretrained", "model": vae}


def encode_video_latent(vae_ctx, video_tensor, device, vae_dtype, non_blocking: bool):
    vae = vae_ctx["model"]
    x = video_tensor.to(device=device, dtype=vae_dtype, non_blocking=non_blocking)
    # Use the model's encode path, which applies Wan's temporal chunking/cache logic.
    mu = vae.encode(x).latent_dist.mean
    latents_mean = torch.tensor(vae.config.latents_mean, dtype=mu.dtype, device=mu.device)
    latents_std = torch.tensor(vae.config.latents_std, dtype=mu.dtype, device=mu.device)
    mu = (mu - latents_mean.view(1, -1, 1, 1, 1)) * (1.0 / latents_std.view(1, -1, 1, 1, 1))
    return mu.float()


def maybe_init_text_encoder(args, device):
    if args.skip_text:
        return None

    text_encoder_dir = args.model_root / "text_encoder"
    tokenizer_dir = args.model_root / "tokenizer"
    if not text_encoder_dir.exists():
        raise FileNotFoundError(
            f"Text encoder dir not found: {text_encoder_dir}. "
            "Pass --skip-text if text embedding is not needed."
        )
    if not tokenizer_dir.exists():
        raise FileNotFoundError(
            f"Tokenizer dir not found: {tokenizer_dir}. "
            "Pass --skip-text if text embedding is not needed."
        )
    tokenizer = T5TokenizerFast.from_pretrained(str(tokenizer_dir))
    text_encoder = UMT5EncoderModel.from_pretrained(
        str(text_encoder_dir),
        torch_dtype=torch.bfloat16,
    ).to(device)
    text_encoder.eval().requires_grad_(False)
    return {
        "mode": "lingbot_pretrained",
        "model": text_encoder,
        "tokenizer": tokenizer,
    }


def encode_text(text_encoder, text: str, device) -> Optional[torch.Tensor]:
    if text_encoder is None:
        return None

    tokenizer = text_encoder["tokenizer"]
    model = text_encoder["model"]
    text_inputs = tokenizer(
        [text],
        padding="max_length",
        max_length=512,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_ids = text_inputs.input_ids.to(device)
    attention_mask = text_inputs.attention_mask.to(device)
    out = model(input_ids, attention_mask).last_hidden_state[0]
    return out.to(torch.bfloat16).cpu()


def build_video_jobs(meta_obj, max_episodes: Optional[int]) -> List[Dict]:
    jobs: List[Dict] = []
    episodes_iter = meta_obj.episodes.values()
    camera_keys = meta_obj.camera_keys
    for idx, ep in enumerate(episodes_iter):
        if max_episodes is not None and idx >= max_episodes:
            break
        episode_index = ep.get("episode_index", idx)
        segments = ep.get("action_config")
        if not segments:
            segments = [{
                "start_frame": 0,
                "end_frame": ep.get("length", 0),
                "action_text": ep.get("tasks", [""])[0] if ep.get("tasks") else "",
            }]

        for cam_key in camera_keys:
            try:
                rel_path = meta_obj.get_video_file_path(episode_index, cam_key)
                video_path = meta_obj.root / rel_path
            except FileNotFoundError as e:
                print(f"[skip] {e}")
                continue
            jobs.append(
                {
                    "episode_index": episode_index,
                    "rel_path": rel_path,
                    "video_path": video_path,
                    "segments": segments,
                }
            )
    return jobs


def ensure_backend_available(backend: str) -> None:
    if backend == "decord" and decord is None:
        raise ImportError("decord is not installed but is required for --video-backend decord.")
    if backend == "pyav" and av is None:
        raise ImportError("av (PyAV) is not installed but is required for --video-backend pyav.")
    if backend == "opencv" and cv2 is None:
        raise ImportError("opencv-python is not installed but is required for --video-backend opencv.")
    if backend == "torchcodec" and VideoDecoder is None:
        raise ImportError(
            "torchcodec is not installed but is required for --video-backend torchcodec."
        )


def resolve_video_backend(backend: str) -> str:
    return backend


def open_video_source(job: Dict, backend: str, video_num_workers: int = 1) -> Dict:
    out = dict(job)
    if backend == "decord":
        vr = decord.VideoReader(str(job["video_path"]))
        out["reader"] = vr
        out["backend"] = "decord"
        out["ori_fps"] = float(vr.get_avg_fps() or 30.0)
        out["total_frames"] = int(len(vr))
        return out

    if backend == "pyav":
        container = av.open(str(job["video_path"]))
        stream = next((s for s in container.streams if s.type == "video"), None)
        if stream is None:
            container.close()
            raise RuntimeError(f"No video stream found: {job['video_path']}")
        if video_num_workers > 1:
            stream.thread_type = "AUTO"
            stream.codec_context.thread_count = int(video_num_workers)
        fps = float(stream.average_rate) if stream.average_rate is not None else 30.0
        out["container"] = container
        out["backend"] = "pyav"
        out["ori_fps"] = fps
        out["total_frames"] = int(stream.frames) if stream.frames else -1
        out["_frame_cache"] = None
        return out

    if backend == "opencv":
        if video_num_workers > 0:
            cv2.setNumThreads(int(video_num_workers))
        cap = cv2.VideoCapture(str(job["video_path"]))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video with OpenCV: {job['video_path']}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or -1)
        out["cap"] = cap
        out["backend"] = "opencv"
        out["ori_fps"] = fps
        out["total_frames"] = total
        out["_frame_cache"] = None
        return out

    if backend == "torchcodec":
        decoder = VideoDecoder(str(job["video_path"]))
        out["decoder"] = decoder
        out["backend"] = "torchcodec"
        out["ori_fps"] = float(getattr(decoder, "metadata").average_fps)
        out["total_frames"] = int(len(decoder))
        return out

    raise ValueError(f"Unsupported video backend: {backend}")


def _decode_all_frames_pyav(video_source: Dict) -> torch.Tensor:
    cached = video_source.get("_frame_cache")
    if cached is not None:
        return cached

    frames = []
    for frame in video_source["container"].decode(video=0):
        frames.append(torch.from_numpy(frame.to_ndarray(format="rgb24")))
    if not frames:
        raise RuntimeError("PyAV decoded 0 frames.")
    stacked = torch.stack(frames, dim=0).contiguous()
    video_source["_frame_cache"] = stacked
    video_source["total_frames"] = int(stacked.shape[0])
    return stacked


def _decode_all_frames_opencv(video_source: Dict) -> torch.Tensor:
    cached = video_source.get("_frame_cache")
    if cached is not None:
        return cached

    cap = video_source["cap"]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(torch.from_numpy(frame))
    if not frames:
        raise RuntimeError("OpenCV decoded 0 frames.")
    stacked = torch.stack(frames, dim=0).contiguous()
    video_source["_frame_cache"] = stacked
    video_source["total_frames"] = int(stacked.shape[0])
    return stacked


def read_selected_frames(video_source: Dict, frame_ids: List[int]) -> torch.Tensor:
    if video_source["backend"] == "decord":
        frames = video_source["reader"].get_batch(frame_ids).asnumpy()
        return torch.from_numpy(frames)

    if video_source["backend"] == "torchcodec":
        decoder = video_source["decoder"]
        # Handle common torchcodec APIs across versions.
        if hasattr(decoder, "get_frames_at"):
            out = decoder.get_frames_at(frame_ids)
            data = out.data if hasattr(out, "data") else out
            return torch.as_tensor(data)
        if hasattr(decoder, "get_batch"):
            out = decoder.get_batch(frame_ids)
            data = out.data if hasattr(out, "data") else out
            return torch.as_tensor(data)
        raise RuntimeError(
            "Unsupported torchcodec VideoDecoder API: expected get_frames_at/get_batch."
        )

    if video_source["backend"] == "pyav":
        all_frames = _decode_all_frames_pyav(video_source)
    elif video_source["backend"] == "opencv":
        all_frames = _decode_all_frames_opencv(video_source)
    else:
        raise ValueError(f"Unknown video backend: {video_source['backend']}")

    max_idx = int(all_frames.shape[0] - 1)
    safe_ids = [min(max(int(i), 0), max_idx) for i in frame_ids]
    return all_frames[safe_ids]


def iter_video_sources(
    video_jobs: List[Dict], backend: str, video_num_workers: int = 1
) -> Iterator[Dict]:
    for job in video_jobs:
        try:
            yield open_video_source(job, backend, video_num_workers=video_num_workers)
        except Exception as e:
            # Some datasets contain broken/non-video files; skip and continue.
            print(f"[skip] failed to open video: {job['video_path']} ({type(e).__name__}: {e})")
            continue


def rel_video_path_to_latent_rel_path(rel_video_path: str) -> Path:
    """
    Convert LeRobot video relative path to latent relative path.
    Example:
      videos/chunk-000/observation.images.cam_high/episode_000000.mp4
      -> chunk-000/observation.images.cam_high/episode_000000.mp4
    """
    p = Path(rel_video_path)
    if p.parts and p.parts[0] == "videos":
        p = Path(*p.parts[1:])
    return p


def discover_repo_ids(repo_root: Path) -> List[str]:
    if not repo_root.exists():
        raise FileNotFoundError(f"repo root not found: {repo_root}")
    repo_ids: List[str] = []
    for child in sorted(repo_root.iterdir()):
        if not child.is_dir():
            continue
        repo_id = child.name
        if (child / "meta" / "info.json").exists():
            repo_ids.append(repo_id)
    return repo_ids


def run_one_repo(args, repo_id: str, repo_root: Optional[Path]) -> None:
    video_backend = resolve_video_backend(args.video_backend)
    ensure_backend_available(video_backend)

    meta_root = (repo_root / repo_id) if repo_root is not None else None
    meta_obj = LeRobotDatasetMetadata(
        repo_id=repo_id,
        root=meta_root,
        force_cache_sync=False,
    )
    device = torch.device(args.device)
    vae_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    vae_ctx = init_vae(args, device, vae_dtype)
    text_encoder = maybe_init_text_encoder(args, device)
    video_jobs = build_video_jobs(meta_obj, args.max_episodes)
    print(f"[info] repo={repo_id}, video backend={video_backend}, jobs={len(video_jobs)}")
    use_non_blocking = device.type == "cuda"
    text_cache: Dict[str, Optional[torch.Tensor]] = {}
    for loaded in tqdm(
        iter_video_sources(
            video_jobs, video_backend, video_num_workers=args.video_num_workers
        ),
        total=len(video_jobs),
        desc="Videos",
    ):
        episode_index = loaded["episode_index"]
        rel_path = loaded["rel_path"]
        ori_fps = loaded["ori_fps"]
        total_frames = loaded["total_frames"]

        pending_items: List[Dict] = []

        def flush_pending():
            if not pending_items:
                return

            max_frames = max(item["video_tensor"].shape[1] for item in pending_items)
            batch_tensors = []
            for item in pending_items:
                vt = item["video_tensor"]
                pad = max_frames - vt.shape[1]
                if pad > 0:
                    vt = torch.cat([vt, vt[:, -1:, :, :].expand(-1, pad, -1, -1)], dim=1)
                batch_tensors.append(vt)
            batch = torch.stack(batch_tensors, dim=0)
            if use_non_blocking:
                batch = batch.pin_memory()

            with torch.no_grad():
                batch_latent = encode_video_latent(
                    vae_ctx, batch, device, vae_dtype, non_blocking=use_non_blocking
                )

            for idx_in_batch, item in enumerate(pending_items):
                frame_ids = item["frame_ids"]
                expected_latent_frames = (len(frame_ids) - 1) // 4 + 1
                latent = batch_latent[idx_in_batch, :, :expected_latent_frames, :, :]
                if latent.shape[1] != expected_latent_frames:
                    raise RuntimeError(
                        "Unexpected latent frame count: "
                        f"got {latent.shape[1]}, expected {expected_latent_frames}"
                    )

                latent_flat = (
                    latent.permute(1, 2, 3, 0)
                    .reshape(-1, latent.shape[0])
                    .to(torch.bfloat16)
                    .cpu()
                )
                text_key = item["text"]
                if text_key not in text_cache:
                    text_cache[text_key] = encode_text(text_encoder, text_key, device)
                text_emb = text_cache[text_key]

                latent_rel_path = rel_video_path_to_latent_rel_path(rel_path)
                save_path = meta_obj.root / "latents" / latent_rel_path
                save_path = save_path.with_name(
                    f"episode_{episode_index:06d}_{item['start']}_{item['end']}.pth"
                )
                save_path.parent.mkdir(parents=True, exist_ok=True)

                payload = {
                    "latent": latent_flat,
                    "latent_num_frames": latent.shape[1],
                    "latent_height": latent.shape[2],
                    "latent_width": latent.shape[3],
                    "video_num_frames": len(frame_ids),
                    "video_height": item["oh"],
                    "video_width": item["ow"],
                    "text_emb": text_emb,
                    "text": item["text"],
                    "frame_ids": frame_ids,
                    "sampled_frame_ids": item["sampled_frame_ids"],
                    "start_frame": item["start"],
                    "end_frame": item["end"],
                    "fps": args.target_fps,
                    "ori_fps": ori_fps,
                }
                torch.save(payload, save_path)

            pending_items.clear()

        for seg in loaded["segments"]:
            start = seg["start_frame"]
            end = seg["end_frame"]
            text = seg.get("action_text", "")

            sampled_frame_ids = sample_frame_ids(
                start, end, ori_fps, args.target_fps, total_frames
            )
            frame_ids = align_frame_ids_for_vae(
                sampled_frame_ids, args.frame_count_align
            )
            if not frame_ids:
                print(
                    f"[warn] empty frame ids for episode {episode_index} segment {start}-{end}"
                )
                continue

            try:
                selected = read_selected_frames(loaded, frame_ids)  # (T, H, W, C)
            except Exception as e:
                print(
                    f"[skip] failed to read frames from {loaded['video_path']} "
                    f"(episode={episode_index}, segment={start}-{end}): "
                    f"{type(e).__name__}: {e}"
                )
                continue
            try:
                video_tensor, (oh, ow) = preprocess_frames(
                    selected, max_area=args.max_area
                )
            except Exception as e:
                print(
                    f"[skip] failed to preprocess frames from {loaded['video_path']} "
                    f"(episode={episode_index}, segment={start}-{end}): "
                    f"{type(e).__name__}: {e}"
                )
                continue
            pending_items.append(
                {
                    "video_tensor": video_tensor,
                    "frame_ids": frame_ids,
                    "sampled_frame_ids": sampled_frame_ids,
                    "start": start,
                    "end": end,
                    "text": text,
                    "oh": oh,
                    "ow": ow,
                }
            )
            if len(pending_items) >= max(1, args.batch_size):
                flush_pending()

        flush_pending()

        # free memory for long runs
        if "reader" in loaded:
            del loaded["reader"]
        if "container" in loaded:
            loaded["container"].close()
            del loaded["container"]
        if "cap" in loaded:
            loaded["cap"].release()
            del loaded["cap"]
        if "_frame_cache" in loaded:
            loaded["_frame_cache"] = None
        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.empty_cache()


def main():
    args = parse_args()

    if args.repo_id is not None:
        repo_ids = [args.repo_id]
        repo_root = args.repo_root
    else:
        repo_root = args.repo_root
        repo_ids = discover_repo_ids(repo_root)
        if not repo_ids:
            raise RuntimeError(f"No datasets found under {repo_root}")

    print(f"[info] single-process mode, device={args.device}, repos={len(repo_ids)}")

    for rid in repo_ids:
        run_one_repo(args, rid, repo_root)


if __name__ == "__main__":
    main()
