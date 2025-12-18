# face-bench

Local benchmarking utilities (batch runs) for the FaceRec pipeline.

This folder is self-contained and uses the copied modules under `ml_models/` and `utils/` plus weights under `model_weights/`.

## Batch verify (one onboard video vs many verify videos)

From `facerec/face-bench/`:

```bash
python scripts/batch_verify_videos.py \
  --onboard "E:/path/to/onboard.mp4" \
  --verify "E:/path/to/verify_videos/" "E:/path/to/one_more.mp4" \
  --quiet \
  --out_csv "bench_results.csv"
```

Frame selection options:

```bash
# First frame (default)
python scripts/batch_verify_videos.py --onboard onboard.mp4 --verify verify_dir --first_frame

# Center frame
python scripts/batch_verify_videos.py --onboard onboard.mp4 --verify verify_dir --center_frame

# Specific frame index
python scripts/batch_verify_videos.py --onboard onboard.mp4 --verify verify_dir --frame_index 0

# Frame at time (seconds)
python scripts/batch_verify_videos.py --onboard onboard.mp4 --verify verify_dir --time_seconds 1.5
```

### Output

Writes a CSV with distances for each verify video vs the onboarding embedding:
- `cosine_distance` (lower is closer)
- `euclidean` (lower is closer)
- `l2_euclidean` (lower is closer)

It also includes `votes_passed` and `match_code` (aligned with the thresholds embedded in `scripts/batch_verify_videos.py`).

## Continuous verification (cumulative averaged anchor)

This runs multiple steps: start with onboard embedding, then iteratively average in each verify embedding (in order), and at every step compare that averaged anchor against all verify videos. Each step is written to its own CSV.

```bash
python scripts/continuous_verify.py \
  --onboard "E:/path/to/onboard.mp4" \
  --verify "E:/path/to/verify_videos/" \
  --quiet \
  --out_dir "results_continuous"
```

CSV files are named `step_00.csv` (onboard only), `step_01.csv` (onboard + first verify), `step_02.csv`, etc. The columns include `anchor_desc` and `anchor_size` so you know which cumulative anchor was used.


