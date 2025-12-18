# Face Verification – Continuous Averaging Experiment

Use this markdown as the source for your Confluence page. Copy/paste the sections below into Confluence and attach/link the referenced assets from the repo.

## Summary
- Goal: evaluate continuous embedding averaging (onboard + progressive verify videos) and its impact on match quality.
- Inputs: `data/onboard/video.mp4` (onboard) and `data/verify/*.mp4` (verify set).
- Outputs: per-step CSVs (`results/step_*.csv`), single-run CSV (`result.csv`), plots exported from `results_analysis.ipynb`.

## Data
- Onboard: `data/onboard/video.mp4`
- Verify videos: `data/verify/*.mp4`
- Generated CSVs: `results/step_00.csv ... step_N.csv`, `result.csv`

## Method
1) Embedding extraction: `scripts/continuous_verify.py` (continuous) and `scripts/batch_verify_videos.py` (single).
2) Frame selection: default first frame (optionally center/index/time).
3) Thresholds: cosine 0.59, euclidean 1.45, l2 1.08; match_code 0 = match, 1 = mismatch.
4) Continuous steps: start with onboard embedding; iteratively average in each verify embedding (in order); at each step compare that anchor against all verify embeddings → write `results/step_XX.csv`.

## How to reproduce
- Repo: `face-bench` (branch `main`).
- Run continuous:
  ```
  python scripts/continuous_verify.py \
    --onboard data/onboard/video.mp4 \
    --verify data/verify \
    --quiet \
    --out_dir results
  ```
- Run single batch:
  ```
  python scripts/batch_verify_videos.py \
    --onboard data/onboard/video.mp4 \
    --verify data/verify \
    --quiet \
    --out_csv result.csv
  ```
- Analysis notebook: `results_analysis.ipynb` (reads `results/step_*.csv` and `result.csv`, produces plots).

## Results and assets
- CSVs: `results/step_00.csv ... step_N.csv`, `result.csv`
- Plots (export from notebook and attach):
  - Average metrics plot (match rate / mean distances vs step) → `average_output.png`
  - Per-file trajectories plot → `filewise_output.png`
  - 2D face map (cosine vs l2, thumbnails) → `output.png`
  - Optional: deltas table snapshots

## Findings (fill in after reviewing plots)
- Example placeholders:
  - Averaging improves/stabilizes match_rate through step X; marginal gains after step Y.
  - Outlier videos: <list stems> show higher cosine/l2 distances across steps.
  - Best anchor size observed at step <N> (anchor_size=<N+1> including onboard).

## Next steps (suggestions)
- Try multi-frame aggregation per video (e.g., median of first 5 frames).
- Adjust thresholds or weighting for cosine vs euclidean.
- Add face-quality filtering before averaging.
- Expand verify set / more lighting/pose variation.

## Links
- Repo: `https://github.com/Namadgi/face-bench` (branch `main`)
- Notebook: `results_analysis.ipynb`
- Scripts: `scripts/continuous_verify.py`, `scripts/batch_verify_videos.py`
- Results: `results/`, `result.csv`


