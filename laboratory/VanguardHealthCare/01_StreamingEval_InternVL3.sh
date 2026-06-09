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

DATA_DIR="" # << Your dataset dir here
EVENT_TYPE="${1:-violence}"     # violence | climbing  (override: ./01_StreamingEval_InternVL3.sh climbing)

# IMPORTANT — --interval 0.09 + --buffer-size 12 are intentional, NOT the script's
# spec defaults (1.0s / 12). The fine-tuned checkpoint above was trained on 12 frames
# at a 0.09s interval, i.e. one ~1-second clip per inference (12 x 0.09s ~= 1.08s).
# Violence is a very short-lived event, so each prediction must look at only ~1 second
# of footage. Keep these two values in sync with how the model was trained.
CUDA_VISIBLE_DEVICES=0 "${PYTHON}" "${SCRIPT_DIR}/01_StreamingEval_InternVL3.py" \
    --data-dir "${DATA_DIR}" \
    --event-type "${EVENT_TYPE}" \
    --model-path "/workspace/sangrak/train_eval_toolkit/ckpts/InternVL3-2B_gangnam_rwf2000_gj_cctv_scvdALL_NOweapon_no_split" \
    --out-dir "${SCRIPT_DIR}/results" \
    --buffer-size 12 \
    --interval 0.09 \
    --overlap-mode min \
    --overlap-threshold 0.5
