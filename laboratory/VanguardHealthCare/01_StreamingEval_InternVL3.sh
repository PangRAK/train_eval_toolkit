#!/usr/bin/env bash
# Streaming InternVL3-2B evaluation on the VanguardHealthCare 2ndPoC dataset.
#
# - Runs deque-based streaming inference over EVERY *.mp4 in --data-dir
#   (one frame sampled per --interval sec, one inference per --buffer-size frames).
# - Grades clips against ONLY the chosen --event-type (its labelled intervals = positives;
#   everything else, incl. normal-only videos, = negatives) and reports per-video + final F1.
#
# Runs from anywhere: it calls the .py sitting next to it by absolute path.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# internvl conda env carries the matching transformers/decord/cv2 stack for InternVL3.
PYTHON="/workspace/sangrak/anaconda3/envs/internvl/bin/python"


# IMPORTANT — --interval 0.09 + --buffer-size 12 are intentional, NOT the script's
# spec defaults (1.0s / 12). The fine-tuned checkpoint above was trained on 12 frames
# at a 0.09s interval, i.e. one ~1-second clip per inference (12 x 0.09s ~= 1.08s).
# Violence is a very short-lived event, so each prediction must look at only ~1 second
# of footage. Keep these two values in sync with how the model was trained.
#
# Set --buffer-size 1 to run single-image inference instead (dynamic tiling + one <image>,
# via InternVL3's image path; see README "Image vs. video inference"). That is for image-style
# checkpoints/ablations — NOT this 12-frame video checkpoint.

DATA_DIR="/workspace/sangrak/01-NAS-sangrak/01_Dataset/01_뱅가드헬스케어/2ndPoC" # << Your dataset dir here
EVENT_TYPE="${1:-violence}"     # violence | climbing  (override: ./01_StreamingEval_InternVL3.sh climbing)
CUDA_VISIBLE_DEVICES=1 "${PYTHON}" "${SCRIPT_DIR}/01_StreamingEval_InternVL3.py" \
    --data-dir "${DATA_DIR}" \
    --event-type "${EVENT_TYPE}" \
    --model-path "/workspace/sangrak/train_eval_toolkit/ckpts/InternVL3-2B_gangnam_rwf2000_gj_cctv_scvdALL_NOweapon_no_split" \
    --out-dir "${SCRIPT_DIR}/results" \
    --buffer-size 12 \
    --interval 0.09 \
    --queue-size 5 \
    --queue-k 3 \
    --overlap-mode min \
    --overlap-threshold 0.5 \
    --unlabeled-as-positive \
    --save-viz


# DATA_DIR="/workspace/sangrak/01-NAS-sangrak/01_Dataset/01_뱅가드헬스케어/2ndPoC" # << Your dataset dir here
# EVENT_TYPE="${1:-climbing}"     # violence | climbing  (override: ./01_StreamingEval_InternVL3.sh climbing)
# CUDA_VISIBLE_DEVICES=1 "${PYTHON}" "${SCRIPT_DIR}/01_StreamingEval_InternVL3.py" \
#     --data-dir "${DATA_DIR}" \
#     --event-type "${EVENT_TYPE}" \
#     --model-path "/workspace/sangrak/train_eval_toolkit/ckpts/InternVL3-2B" \
#     --out-dir "${SCRIPT_DIR}/results" \
#     --buffer-size 1 \
#     --interval 0.09 \
#     --queue-size 5 \
#     --queue-k 3 \
#     --overlap-mode min \
#     --overlap-threshold 0.5 \
#     --unlabeled-as-positive \
#     --save-viz

# --queue-size / --queue-k: alarm smoothing over a sliding queue of recent window predictions.
# The model still predicts per ~1s window, but the alarm only fires when >= --queue-k of the last
# --queue-size windows were positive (here: >=3 of the last 5). The queue slides one window per
# inference (oldest out, newest in) and resets per video. Grading + viz use this alarm decision,
# not the raw window prediction. Set --queue-size 1 --queue-k 1 for the old single-window behaviour.
#
# --save-viz: also write an annotated .mp4 per video to results/<ts>_<event>_viz/ — the sampled
# frames with a red alert border + event label overlaid on every clip the model flags. Add
# --viz-only-alarms to keep just the videos where the model fired. Needs ffmpeg on PATH.

# --unlabeled-as-positive: this DATA_DIR (clips/violence) holds only target-event clips with
# NO label JSONs, so every clip is graded as a ground-truth positive (correct detection = TP,
# miss = FN). Drop this flag if you point DATA_DIR at a normally-labelled folder.