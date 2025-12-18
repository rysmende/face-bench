from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
from PIL import Image
import torch


def _bench_root() -> Path:
    # This script lives in: <face-bench>/scripts/batch_verify_videos.py
    return Path(__file__).resolve().parents[1]


def _add_face_bench_to_syspath() -> None:
    """
    Ensure imports like `from ml_models...` and `from utils...` work regardless of CWD.
    """
    root = _bench_root()
    sys.path.insert(0, str(root))


# Match the thresholds used by the service logic.
COSINE_THRESHOLD: float = 0.59
EUCLIDEAN_THRESHOLD: float = 1.45
EUCLIDEAN_L2_THRESHOLD: float = 1.08
ACCEPTED_THRESHOLD: int = 2


def _configure_videoio_logging(quiet: bool) -> None:
    """
    Reduce noisy stderr logs from OpenCV's video backends (often FFmpeg demuxer warnings).
    """
    if not quiet:
        return

    # Best-effort: different builds honor different knobs.
    os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

    # OpenCV python bindings expose a log-level setter in many versions.
    try:
        # Values are not exposed as constants in this build; use a conservative numeric level.
        # 0=Verbose, 1=Debug, 2=Info, 3=Warning, 4=Error, 5=Fatal, 6=Silent (common mapping)
        cv2.setLogLevel(4)
    except Exception:
        pass


@dataclass(frozen=True)
class FrameSpec:
    # Exactly one should be set.
    first: bool = True
    center: bool = False
    frame_index: Optional[int] = None
    time_seconds: Optional[float] = None


def _extract_frame_bgr(video_path: Path, spec: FrameSpec) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        # If first-frame mode: avoid seeking entirely for max compatibility.
        if spec.first and spec.frame_index is None and spec.time_seconds is None and not spec.center:
            # Some videos may return an empty/None frame initially; try a few reads.
            for _ in range(30):
                ok, frame = cap.read()
                if ok and frame is not None:
                    return frame
            raise RuntimeError(f"Could not read first frame from video: {video_path}")

        target_idx: Optional[int] = None
        if spec.frame_index is not None:
            target_idx = max(0, int(spec.frame_index))
        elif spec.time_seconds is not None:
            if fps <= 0:
                # FPS unknown; fall back to center if possible
                target_idx = (frame_count // 2) if frame_count > 0 else 0
            else:
                target_idx = max(0, int(round(spec.time_seconds * fps)))
        else:
            # center
            target_idx = (frame_count // 2) if frame_count > 0 else 0

        # Seek if supported
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(target_idx))
        ok, frame = cap.read()
        if ok and frame is not None:
            return frame

        # Fallback: rewind and iterate
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0.0)
        cur = 0
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if cur >= target_idx:
                return frame
            cur += 1

        raise RuntimeError(f"Could not read frame (idx={target_idx}) from video: {video_path}")
    finally:
        cap.release()


def _bgr_to_pil(frame_bgr: np.ndarray) -> Image.Image:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


def _iter_videos(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    exts = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
    for p in sorted(path.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def _load_facerec(weights_path: Path, device: torch.device):
    _add_face_bench_to_syspath()
    from ml_models.facerec import FaceRec  # type: ignore

    model = FaceRec()
    state = torch.load(str(weights_path), map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def _embed_video(model, video_path: Path, device: torch.device, frame_spec: FrameSpec) -> tuple[Optional[torch.Tensor], int]:
    frame_bgr = _extract_frame_bgr(video_path, frame_spec)
    img = _bgr_to_pil(frame_bgr)
    emb, code = model(img, device)
    return emb, code


def main() -> int:
    default_weights = _bench_root() / "model_weights" / "facerec.pth"

    parser = argparse.ArgumentParser(
        description="Batch verify: compare one onboarding video vs many verification videos using FaceRec embeddings.",
    )
    parser.add_argument("--onboard", required=True, help="Path to onboarding video")
    parser.add_argument(
        "--verify",
        required=True,
        nargs="+",
        help="One or more video paths and/or directories containing videos",
    )
    parser.add_argument(
        "--weights",
        default=str(default_weights),
        help="Path to model_weights/facerec.pth",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Inference device",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress noisy OpenCV video decoding warnings (recommended)",
    )

    frame_group = parser.add_mutually_exclusive_group()
    frame_group.add_argument("--first_frame", action="store_true", help="Use the first frame (default)")
    frame_group.add_argument("--center_frame", action="store_true", help="Use the center frame")
    frame_group.add_argument("--frame_index", type=int, help="Use a specific 0-based frame index")
    frame_group.add_argument("--time_seconds", type=float, help="Use a frame at given time (seconds)")

    parser.add_argument(
        "--out_csv",
        default="bench_results.csv",
        help="Output CSV path (relative to current working dir unless absolute)",
    )
    args = parser.parse_args()
    _configure_videoio_logging(bool(args.quiet))

    onboard_path = Path(args.onboard)
    weights_path = Path(args.weights)
    out_csv = Path(args.out_csv)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if args.frame_index is not None:
        frame_spec = FrameSpec(first=False, center=False, frame_index=args.frame_index)
    elif args.time_seconds is not None:
        frame_spec = FrameSpec(first=False, center=False, time_seconds=args.time_seconds)
    elif args.center_frame:
        frame_spec = FrameSpec(first=False, center=True)
    else:
        # default: first frame for max compatibility across OpenCV backends
        frame_spec = FrameSpec(first=True, center=False)

    if not onboard_path.exists():
        raise FileNotFoundError(f"Onboard video not found: {onboard_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    model = _load_facerec(weights_path, device)

    # Import metrics after sys.path is set
    from utils.distance_metrics import (  # type: ignore
        get_cosine_similarity,
        get_euclidean_distance,
        get_l2_euclidean_distance,
    )

    cosine_thr, euclid_thr, l2_thr, accepted_threshold = (
        COSINE_THRESHOLD,
        EUCLIDEAN_THRESHOLD,
        EUCLIDEAN_L2_THRESHOLD,
        ACCEPTED_THRESHOLD,
    )

    onboard_emb, onboard_code = _embed_video(model, onboard_path, device, frame_spec)
    if onboard_code != 0 or onboard_emb is None:
        raise RuntimeError(
            f"Failed to embed onboarding video (code={onboard_code}). "
            f"Fix the onboarding video/frame or change frame selection."
        )

    rows: list[dict[str, object]] = []
    for verify_arg in args.verify:
        for verify_path in _iter_videos(Path(verify_arg)):
            try:
                emb, code = _embed_video(model, verify_path, device, frame_spec)
                if code != 0 or emb is None:
                    rows.append(
                        {
                            "verify_video": str(verify_path),
                            "status": "embed_failed",
                            "embed_code": int(code),
                            "cosine_distance": "",
                            "euclidean": "",
                            "l2_euclidean": "",
                            "votes_passed": "",
                            "match_code": "",
                        }
                    )
                    continue

                # Metrics: smaller is better (these functions return distances)
                cosine = get_cosine_similarity(emb, onboard_emb).item()
                euclid = get_euclidean_distance(emb, onboard_emb).item()
                l2 = get_l2_euclidean_distance(emb, onboard_emb).item()

                votes = int(cosine <= cosine_thr) + int(euclid <= euclid_thr) + int(l2 <= l2_thr)
                match = int(votes >= accepted_threshold)
                code_out = int(not match)  # 0 success, 1 mismatch (matches VerifyService convention)

                rows.append(
                    {
                        "verify_video": str(verify_path),
                        "status": "ok",
                        "embed_code": 0,
                        "cosine_distance": float(cosine),
                        "euclidean": float(euclid),
                        "l2_euclidean": float(l2),
                        "votes_passed": votes,
                        "match_code": code_out,
                    }
                )
            except Exception as e:
                rows.append(
                    {
                        "verify_video": str(verify_path),
                        "status": "error",
                        "embed_code": "",
                        "cosine_distance": "",
                        "euclidean": "",
                        "l2_euclidean": "",
                        "votes_passed": "",
                        "match_code": "",
                        "error": str(e),
                    }
                )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "verify_video",
        "status",
        "embed_code",
        "cosine_distance",
        "euclidean",
        "l2_euclidean",
        "votes_passed",
        "match_code",
        "error",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            # ensure all fields exist
            out = {k: r.get(k, "") for k in fieldnames}
            w.writerow(out)

    print(f"Wrote {len(rows)} rows to {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


