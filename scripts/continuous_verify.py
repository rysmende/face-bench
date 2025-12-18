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
    return Path(__file__).resolve().parents[1]


def _add_face_bench_to_syspath() -> None:
    root = _bench_root()
    sys.path.insert(0, str(root))


# Thresholds aligned with the service.
COSINE_THRESHOLD: float = 0.59
EUCLIDEAN_THRESHOLD: float = 1.45
EUCLIDEAN_L2_THRESHOLD: float = 1.08
ACCEPTED_THRESHOLD: int = 2


def _configure_videoio_logging(quiet: bool) -> None:
    if not quiet:
        return
    os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
    try:
        cv2.setLogLevel(4)  # Error and above
    except Exception:
        pass


@dataclass(frozen=True)
class FrameSpec:
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

        if spec.first and spec.frame_index is None and spec.time_seconds is None and not spec.center:
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
                target_idx = (frame_count // 2) if frame_count > 0 else 0
            else:
                target_idx = max(0, int(round(spec.time_seconds * fps)))
        else:
            target_idx = (frame_count // 2) if frame_count > 0 else 0

        cap.set(cv2.CAP_PROP_POS_FRAMES, float(target_idx))
        ok, frame = cap.read()
        if ok and frame is not None:
            return frame

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


def _compare(emb_anchor: torch.Tensor, emb_target: torch.Tensor):
    from utils.distance_metrics import (  # type: ignore
        get_cosine_similarity,
        get_euclidean_distance,
        get_l2_euclidean_distance,
    )
    cosine = get_cosine_similarity(emb_target, emb_anchor).item()
    euclid = get_euclidean_distance(emb_target, emb_anchor).item()
    l2 = get_l2_euclidean_distance(emb_target, emb_anchor).item()
    votes = int(cosine <= COSINE_THRESHOLD) + int(euclid <= EUCLIDEAN_THRESHOLD) + int(l2 <= EUCLIDEAN_L2_THRESHOLD)
    match = int(votes >= ACCEPTED_THRESHOLD)
    code_out = int(not match)  # 0 success, 1 mismatch
    return cosine, euclid, l2, votes, code_out


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
        "anchor_desc",
        "anchor_size",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            out = {k: r.get(k, "") for k in fieldnames}
            w.writerow(out)


def main() -> int:
    default_weights = _bench_root() / "model_weights" / "facerec.pth"

    parser = argparse.ArgumentParser(
        description=(
            "Continuous verification: progressively average embeddings "
            "(onboard + first verify + second verify + ...) and at each step "
            "compare that anchor against all verify videos, writing a CSV per step."
        )
    )
    parser.add_argument("--onboard", required=True, help="Path to onboarding video")
    parser.add_argument("--verify", required=True, nargs="+", help="One or more verify video paths and/or directories")
    parser.add_argument("--weights", default=str(default_weights), help="Path to model_weights/facerec.pth")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Inference device")
    parser.add_argument("--out_dir", default="results_continuous", help="Directory to write per-step CSVs")
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

    parser.add_argument("--out_prefix", default="step", help="Prefix for CSV files (default: step)")

    args = parser.parse_args()

    _configure_videoio_logging(bool(args.quiet))

    onboard_path = Path(args.onboard)
    weights_path = Path(args.weights)
    out_dir = Path(args.out_dir)

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
        frame_spec = FrameSpec(first=True, center=False)

    if not onboard_path.exists():
        raise FileNotFoundError(f"Onboard video not found: {onboard_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    # Collect verify video paths
    verify_paths: list[Path] = []
    for v in args.verify:
        verify_paths.extend(list(_iter_videos(Path(v))))
    if not verify_paths:
        raise RuntimeError("No verify videos found.")

    model = _load_facerec(weights_path, device)

    # Embed onboard
    onboard_emb, onboard_code = _embed_video(model, onboard_path, device, frame_spec)
    if onboard_code != 0 or onboard_emb is None:
        raise RuntimeError(f"Failed to embed onboarding video (code={onboard_code}).")

    # Embed verify set
    verify_embs: list[tuple[Path, Optional[torch.Tensor], int]] = []
    for vp in verify_paths:
        try:
            emb, code = _embed_video(model, vp, device, frame_spec)
            verify_embs.append((vp, emb, code))
        except Exception as e:
            verify_embs.append((vp, None, -1))
            print(f"[warn] Failed to embed {vp}: {e}")

    # Anchor accumulation
    anchor_sum = onboard_emb.clone()
    anchor_count = 1

    # Steps:
    # step 0: onboard only
    # step i: onboard + first i verify embeddings that succeeded (in order)
    for step_idx in range(0, len(verify_embs) + 1):
        if step_idx > 0:
            vp, emb, code = verify_embs[step_idx - 1]
            if code == 0 and emb is not None:
                anchor_sum = anchor_sum + emb
                anchor_count += 1

        anchor = anchor_sum / anchor_count
        anchor_desc = "onboard" if step_idx == 0 else f"onboard+first_{step_idx}_verify"

        rows: list[dict[str, object]] = []
        for vp, emb, code in verify_embs:
            if code != 0 or emb is None:
                rows.append(
                    {
                        "verify_video": str(vp),
                        "status": "embed_failed",
                        "embed_code": int(code),
                        "cosine_distance": "",
                        "euclidean": "",
                        "l2_euclidean": "",
                        "votes_passed": "",
                        "match_code": "",
                        "error": "",
                        "anchor_desc": anchor_desc,
                        "anchor_size": anchor_count,
                    }
                )
                continue

            try:
                cosine, euclid, l2, votes, match_code = _compare(anchor, emb)
                rows.append(
                    {
                        "verify_video": str(vp),
                        "status": "ok",
                        "embed_code": 0,
                        "cosine_distance": float(cosine),
                        "euclidean": float(euclid),
                        "l2_euclidean": float(l2),
                        "votes_passed": votes,
                        "match_code": match_code,
                        "error": "",
                        "anchor_desc": anchor_desc,
                        "anchor_size": anchor_count,
                    }
                )
            except Exception as e:
                rows.append(
                    {
                        "verify_video": str(vp),
                        "status": "error",
                        "embed_code": "",
                        "cosine_distance": "",
                        "euclidean": "",
                        "l2_euclidean": "",
                        "votes_passed": "",
                        "match_code": "",
                        "error": str(e),
                        "anchor_desc": anchor_desc,
                        "anchor_size": anchor_count,
                    }
                )

        csv_name = f"{args.out_prefix}_{step_idx:02d}.csv"
        _write_csv(out_dir / csv_name, rows)
        print(f"Wrote step {step_idx} -> {out_dir / csv_name} (anchor_size={anchor_count})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


