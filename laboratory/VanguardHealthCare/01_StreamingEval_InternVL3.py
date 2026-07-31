#!/usr/bin/env python
"""Streaming (deque-based) InternVL3-2B evaluation for the VanguardHealthCare 2ndPoC dataset.

What this does
--------------
1. Walks a data folder of CCTV clips (``*.mp4`` + matching ``*.json`` label files).
2. For **every** video (regardless of which event class it carries), reads frames
   sequentially with OpenCV — exactly like
   ``perception_models/laboratory/Distribution/01_TuningFree_v2.py`` — and samples
   one frame every ``--interval`` seconds into a deque. When the deque holds
   ``--buffer-size`` frames it runs **one** InternVL3 inference over those frames and
   then clears the deque (non-overlapping sliding window).
3. Each inference produces a per-window binary prediction for the chosen
   ``--event-type`` (e.g. ``violence`` / ``climbing``). To suppress single-window
   flicker, the **alarm** is a sliding-queue vote over the last ``--queue-size``
   windows: every new window pushes its prediction and drops the oldest, and the
   alarm fires only when at least ``--queue-k`` of the queued windows predicted the
   event. The alarm decision — not the raw per-window prediction — is what gets
   graded and visualized. (``--queue-size 1 --queue-k 1`` reproduces the original
   single-window behaviour.)
4. Each clip is graded against the ground-truth event intervals of that same event
   type taken from the ``.json`` label (``annotations_frame`` + ``annotations_class``):
   a clip is a ground-truth positive when its frame window overlaps a target-class
   event interval by at least ``--overlap-threshold`` (overlap-ratio criterion, see
   ``overlap_ratio``).
5. Per-video and final (corpus-wide) precision / recall / F1 are reported and saved.

Scoring scope (per the spec)
----------------------------
Inference is run on *all* videos in the folder, but grading is done **only** for the
single ``--event-type`` passed on the command line. Videos with no events, or with
events of a *different* class, contribute only negatives for that run — so a model
that fires on them counts as a false positive.

Unlabeled folders (``--unlabeled-as-positive``)
-----------------------------------------------
When a data folder is curated to contain *only* the target event but ships **without**
per-clip label JSONs (e.g. ``.../clips/violence/`` with bare ``*.mp4``), the default
"no label -> all negatives" rule wrongly scores every correct detection as a false
positive. Pass ``--unlabeled-as-positive`` to treat each video that has no label JSON as
all-positive for ``--event-type`` (every clip is a GT positive: a correct detection is a
TP, a miss is an FN). Videos that *do* have a label JSON are still graded by overlap.

Model setting (fixed by the spec)
----------------------------------
InternVL3-2B · NUM_SEGMENTS=12 · buffer_size=12 · interval=1s · max_num=1 (tiles/frame)
· max_new_tokens=15 · bfloat16 · single GPU · 448x448 input.

Image vs. video inference
-------------------------
With ``--buffer-size N>1`` each clip is N sampled frames fed as a video: one 448x448 tile per
frame, addressed by a ``FrameK: <image>`` prefix. With ``--buffer-size 1`` the script instead
takes InternVL3's single-image path: the one frame is split into up to ``--image-max-num``
dynamic tiles (default 12) plus a thumbnail and addressed by a single ``<image>`` — i.e. genuine
image inference, exactly as the InternVL3 model card's ``load_image`` does it (same
``dynamic_preprocess``, ``use_thumbnail=True``).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
from collections import deque

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import yaml
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

# Make the sibling ``utils/`` importable regardless of the caller's CWD (the launcher
# invokes this script by absolute path, often from the repo root).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.utils import VideoFromPIL, draw_alarm  # noqa: E402

# ---------------------------------------------------------------------------
# Model / pipeline constants (fixed by the task spec)
# ---------------------------------------------------------------------------
NUM_SEGMENTS = 12              # frames fed to the model per inference
BUFFER_SIZE = 12               # frames collected per clip (must equal NUM_SEGMENTS)
INTERVAL_SEC = 1               # sampling interval (sec) for filling the buffer
FRAME_PER_TILE_MAX_NUM = 1     # video path: tiles per frame (1 = no tiling, per InternVL3 video docs)
IMAGE_TILE_MAX_NUM = 12        # image path (buffer-size=1): max dynamic tiles for the single image
QUEUE_SIZE = 5                 # alarm vote: # of recent per-window predictions kept in the sliding queue
QUEUE_K = 3                    # alarm fires when >= QUEUE_K of the last QUEUE_SIZE windows are positive
MAX_NEW_TOKEN = 15             # generated tokens
INPUT_SIZE = 448               # input resolution (448x448)
MODEL_INF_DATA_TYPE = torch.bfloat16

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "ckpts", "InternVL3-2B")
)

# ---------------------------------------------------------------------------
# Event specifications.
#   prompt           : instruction appended after the per-frame <image> prefix.
#                      The text lives in the sibling ``prompts.yaml`` (one block
#                      scalar per event type, keyed by event name) so prompts can
#                      be edited without touching this script.
#   positive_aliases : substrings in the model output that mean "event present".
#   class_names      : normalized class strings (annotations_class) that count as
#                      a ground-truth positive for this event type.
# ---------------------------------------------------------------------------
PROMPTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts.yaml")


def load_prompts(path: str = PROMPTS_PATH) -> dict:
    """Load ``{event_type: prompt}`` from the sibling YAML file (trailing newline stripped)."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return {k: str(v).strip() for k, v in data.items()}


_PROMPTS = load_prompts()


EVENT_SPECS = {
    "violence": {
        "prompt": _PROMPTS["violence"],
        "positive_aliases": ["violence", "violent", "fight", "fighting", "abnormal"],
        "class_names": {"violence"},
    },
    "climbing": {
        "prompt": _PROMPTS["climbing"],
        "positive_aliases": ["climb", "climbing", "abnormal"],
        "class_names": {"climbing up", "climbing", "climb up"},
    },
}

NORMAL_ALIASES = ["normal", "none", "no event", "nothing"]


# ===========================================================================
# InternVL3 frame preprocessing (matches src/utils/internvl_perprocess.py)
# ===========================================================================
def build_transform(input_size: int = INPUT_SIZE) -> T.Compose:
    """torchvision transform matching InternVL pre-training (resize -> tensor -> normalize)."""
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        tgt_ar = ratio[0] / ratio[1]
        diff = abs(aspect_ratio - tgt_ar)
        if diff < best_ratio_diff or (
            diff == best_ratio_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]
        ):
            best_ratio_diff = diff
            best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=FRAME_PER_TILE_MAX_NUM, image_size=INPUT_SIZE, use_thumbnail=False):
    """Split an image into <= max_num tiles of image_size (InternVL spec). max_num=1 -> single tile."""
    ow, oh = image.size
    aspect_ratio = ow / oh
    target_ratios = sorted(
        {
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if min_num <= i * j <= max_num
        },
        key=lambda x: x[0] * x[1],
    )
    ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, ow, oh, image_size)
    tw, th = image_size * ratio[0], image_size * ratio[1]
    blocks = ratio[0] * ratio[1]
    resized = image.resize((tw, th))
    tiles = [
        resized.crop(
            (
                (idx % (tw // image_size)) * image_size,
                (idx // (tw // image_size)) * image_size,
                ((idx % (tw // image_size)) + 1) * image_size,
                ((idx // (tw // image_size)) + 1) * image_size,
            )
        )
        for idx in range(blocks)
    ]
    if use_thumbnail and blocks != 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles


def frames_to_pixel_values(pil_frames, transform, max_num=FRAME_PER_TILE_MAX_NUM, image_size=INPUT_SIZE):
    """Turn a list of PIL frames into (pixel_values, num_patches_list) for model.chat().

    With max_num=1 every frame yields a single 448x448 tile, so num_patches_list = [1, 1, ...].
    """
    pixel_values_list, num_patches_list = [], []
    for img in pil_frames:
        tiles = dynamic_preprocess(img, image_size=image_size, use_thumbnail=False, max_num=max_num)
        pv = torch.stack([transform(tile) for tile in tiles])
        num_patches_list.append(pv.shape[0])
        pixel_values_list.append(pv)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list


def image_to_pixel_values(pil_image, transform, max_num=IMAGE_TILE_MAX_NUM, image_size=INPUT_SIZE):
    """Single-image InternVL3 path (buffer-size=1): dynamic tiling + thumbnail.

    Mirrors InternVL3's own ``load_image`` (``use_thumbnail=True``, up to ``max_num`` tiles) by
    reusing the same ``dynamic_preprocess`` the model card ships — i.e. genuine image inference,
    not the one-tile-per-frame video path. The image is split into <= ``max_num`` tiles (plus a
    resized thumbnail when it tiles), so ``num_patches_list = [n_tiles]`` and the prompt carries a
    single ``<image>`` placeholder.
    """
    tiles = dynamic_preprocess(pil_image, image_size=image_size, use_thumbnail=True, max_num=max_num)
    pixel_values = torch.stack([transform(tile) for tile in tiles])
    return pixel_values, [pixel_values.shape[0]]


# ===========================================================================
# Prediction parsing
# ===========================================================================
def parse_prediction(pred_str: str, spec: dict):
    """Map a raw model answer to 1 (event), 0 (normal), or None (unparseable).

    Handles a bare keyword, a ``category`` field, or a small JSON object — robust to the
    truncation that a 15-token budget causes.
    """
    if not pred_str:
        return None
    text = pred_str.strip().lower()

    # Prefer an explicit "category": "..." field if the model emitted JSON-ish output.
    cat_match = re.search(r'["\']?category["\']?\s*:\s*["\']?([a-z _-]+)', text)
    scope = cat_match.group(1).strip() if cat_match else text

    has_pos = any(alias in scope for alias in spec["positive_aliases"])
    has_neg = any(alias in scope for alias in NORMAL_ALIASES)

    if has_pos and not has_neg:
        return 1
    if has_neg and not has_pos:
        return 0
    if has_pos and has_neg:
        # Ambiguous (both words present) -> fall back to whichever appears first.
        pos_at = min((scope.find(a) for a in spec["positive_aliases"] if a in scope), default=10**9)
        neg_at = min((scope.find(a) for a in NORMAL_ALIASES if a in scope), default=10**9)
        return 1 if pos_at < neg_at else 0
    return None


def extract_description(pred_raw: str):
    """Best-effort pull of the ``"description"`` field from the raw VQA output (for overlays).

    The 15-token budget often truncates the JSON, so this just grabs whatever text
    follows ``description:`` and may itself be cut off. Returns None if absent.
    """
    if not pred_raw:
        return None
    m = re.search(r'["\']?description["\']?\s*:\s*["\']([^"\']+)', pred_raw)
    return m.group(1).strip() if m else None


# ===========================================================================
# Ground-truth handling
# ===========================================================================
def load_target_events(json_path: str, class_names: set) -> list:
    """Return [(start_frame, end_frame), ...] for events whose class matches the run's event type.

    Reads ``annotations_frame`` (S{n}/E{n} frame indices) and ``annotations_class`` (E{n} -> class).
    """
    if not os.path.isfile(json_path):
        return []
    with open(json_path, "r") as f:
        data = json.load(f)
    frames = data.get("annotations_frame", {}) or {}
    classes = data.get("annotations_class", {}) or {}

    events = []
    n = 1
    while f"S{n}" in frames and f"E{n}" in frames:
        cls = str(classes.get(f"E{n}", "")).strip().lower()
        if cls in class_names:
            s = int(round(float(frames[f"S{n}"])))
            e = int(round(float(frames[f"E{n}"])))
            events.append((min(s, e), max(s, e)))
        n += 1
    return events


def overlap_ratio(clip_window, event, mode: str = "min") -> float:
    """Overlap ratio between a clip frame-window [f0,f1] and an event interval [s,e].

    mode:
      min   -> intersection / min(clip_len, event_len)   (overlap coefficient; default)
      clip  -> intersection / clip_len
      event -> intersection / event_len
      iou   -> intersection / union
    Lengths are inclusive frame counts.
    """
    f0, f1 = clip_window
    s, e = event
    inter = max(0, min(f1, e) - max(f0, s) + 1)
    if inter <= 0:
        return 0.0
    clip_len = f1 - f0 + 1
    evt_len = e - s + 1
    if mode == "clip":
        denom = clip_len
    elif mode == "event":
        denom = evt_len
    elif mode == "iou":
        denom = clip_len + evt_len - inter
    else:  # "min"
        denom = min(clip_len, evt_len)
    return float(inter / denom) if denom > 0 else 0.0


def clip_is_positive(clip_window, events, mode: str, threshold: float) -> bool:
    """True if the clip overlaps any target event by >= threshold under the chosen ratio mode."""
    return any(overlap_ratio(clip_window, ev, mode) >= threshold for ev in events)


def best_overlap(clip_window, events, mode: str):
    """Return (best_ratio, best_event) for the target event with the highest overlap (or (0.0, None))."""
    best_ratio, best_event = 0.0, None
    for ev in events:
        r = overlap_ratio(clip_window, ev, mode)
        if r > best_ratio:
            best_ratio, best_event = r, ev
    return best_ratio, best_event


# ===========================================================================
# Metrics
# ===========================================================================
def binary_metrics(y_true, y_pred) -> dict:
    """Precision / recall / F1 / accuracy + confusion counts for binary labels (1=positive)."""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    n = len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / n if n > 0 else 0.0
    return {
        "n_clips": n,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy,
    }


# ===========================================================================
# Model wrapper
# ===========================================================================
class InternVL3Streamer:
    def __init__(self, model_path: str, device: str = "cuda", image_max_num: int = IMAGE_TILE_MAX_NUM):
        print(f"[INFO] Loading InternVL3 from: {model_path}")
        self.model = (
            AutoModel.from_pretrained(
                model_path,
                torch_dtype=MODEL_INF_DATA_TYPE,
                low_cpu_mem_usage=True,
                use_flash_attn=False,
                trust_remote_code=True,
            )
            .eval()
            .to(device)
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        self.device = device
        self.transform = build_transform(INPUT_SIZE)
        self.image_max_num = image_max_num
        self.generation_config = dict(max_new_tokens=MAX_NEW_TOKEN, do_sample=False)
        print("[INFO] Model loaded.")

    @torch.inference_mode()
    def infer_clip(self, pil_frames, prompt: str):
        """Run one clip. Returns (response_text, timing) with timing = {preprocess, inference, total} sec."""
        cuda = torch.cuda.is_available()
        t0 = time.time()
        if len(pil_frames) == 1:
            # IMAGE inference (buffer-size=1): InternVL3's single-image path — one image split
            # into <= image_max_num dynamic tiles, addressed by a single "<image>" placeholder.
            pixel_values, num_patches_list = image_to_pixel_values(
                pil_frames[0], self.transform, max_num=self.image_max_num
            )
            prefix = "<image>\n"
        else:
            # VIDEO inference: one 448x448 tile per frame, one "<image>" per frame. The count MUST
            # equal len(num_patches_list), so build the prefix from the buffered frame count.
            pixel_values, num_patches_list = frames_to_pixel_values(pil_frames, self.transform)
            prefix = "".join(f"Frame{i + 1}: <image>\n" for i in range(len(num_patches_list)))
        pixel_values = pixel_values.to(self.device, dtype=MODEL_INF_DATA_TYPE)
        if cuda:
            torch.cuda.synchronize()
        t1 = time.time()
        question = prefix + prompt
        response = self.model.chat(
            self.tokenizer,
            pixel_values,
            question,
            self.generation_config,
            num_patches_list=num_patches_list,
            verbose=False,
        )
        if cuda:
            torch.cuda.synchronize()
        t2 = time.time()
        timing = {"preprocess": t1 - t0, "inference": t2 - t1, "total": t2 - t0}
        return response, timing


# ===========================================================================
# Per-video streaming inference + grading
# ===========================================================================
def process_video(streamer, video_path, json_path, spec, args, global_run=None, viz_dir=None):
    """Stream one video, run clip inferences, grade them, and return (records, video_metrics).

    Each window inference yields a raw per-window prediction (0/1). The graded decision is the
    ALARM: a sliding-queue vote over the last ``args.queue_size`` windows that fires when
    ``>= args.queue_k`` of them were positive. The queue is per-video (reset for each video) so
    alarm state never leaks across clips of different videos.

    global_run, if given, is a shared {tp,fp,fn,tn} dict accumulated across ALL videos so a
    corpus-wide running F1 can be printed at every clip.

    viz_dir, if given, enables alarm visualization: each clip's sampled frames are written to
    ``<viz_dir>/<video>_<event>.mp4`` with a red alert border + event label overlaid on the
    frames of clips predicted positive (see ``utils.utils.draw_alarm``).
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps is None or fps <= 0:
        print(f"[WARN] {os.path.basename(video_path)}: invalid fps -> skipped")
        cap.release()
        return [], None

    step = max(1, int(round(fps * args.interval)))  # frames between sampled frames
    events = load_target_events(json_path, spec["class_names"])
    # A video with NO label JSON contributes no events, so by default every clip is a GT
    # negative. With --unlabeled-as-positive we instead assume the whole (unlabeled) video IS
    # the target event — the folder is curated to contain only it — and grade every clip as a
    # GT positive. Videos that DO have a label JSON are graded by overlap as usual.
    has_label = os.path.isfile(json_path)
    force_positive = args.unlabeled_as_positive and not has_label
    pos_name = args.event_type
    n_sampled = (total_frames + step - 1) // step if total_frames > 0 else 0
    est_total_clips = n_sampled // args.buffer_size  # rough "i/~N" denominator

    buf = deque()  # (frame_idx, PIL.Image)
    pred_queue = deque(maxlen=args.queue_size)  # recent per-window predictions (0/1) for alarm voting
    records = []
    parse_fail = 0
    frame_idx = -1
    clip_idx = 0
    run = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}  # running confusion counts (this video)
    pre_times, inf_times, tot_times = [], [], []

    # Alarm visualization: a writer is created lazily on the first completed clip so a video
    # too short to fill one buffer produces no file. Sampled frames play back at 1/interval
    # (real time) unless --viz-fps overrides it.
    viz_writer, had_alarm = None, False
    viz_path = (
        os.path.join(viz_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}_{pos_name}.mp4")
        if viz_dir else None
    )
    viz_fps = args.viz_fps if args.viz_fps > 0 else max(1.0, 1.0 / args.interval)

    while True:
        frame_idx += 1
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step != 0:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        buf.append((frame_idx, Image.fromarray(rgb)))

        if len(buf) < args.buffer_size:
            continue

        # Buffer full -> one inference over the buffered frames, then empty the buffer.
        clip_idx += 1
        window = (buf[0][0], buf[-1][0])
        pred_raw, timing = streamer.infer_clip([p for _, p in buf], spec["prompt"])
        pre_times.append(timing["preprocess"])
        inf_times.append(timing["inference"])
        tot_times.append(timing["total"])

        pred = parse_prediction(pred_raw, spec)
        if pred is None:
            parse_fail += 1
            pred_label = 1 if args.unparsed_positive else 0
        else:
            pred_label = pred

        # Per-window prediction (raw, 0/1) feeds the sliding alarm queue. The ALARM — the
        # decision that gets graded/visualized — fires when >= queue_k of the last queue_size
        # windows were positive. Before the queue fills it votes over however many it holds.
        window_pred = pred_label
        pred_queue.append(window_pred)
        queue_pos = sum(pred_queue)
        alarm_label = 1 if queue_pos >= args.queue_k else 0

        if force_positive:
            # Unlabeled video assumed to be the target event -> clip is a GT positive.
            gt_label = 1
            match_ratio, match_ev = None, None
        else:
            match_ratio, match_ev = best_overlap(window, events, args.overlap_mode)
            gt_label = 1 if match_ratio >= args.overlap_threshold else 0

        # confusion bucket + running tally (graded on the ALARM decision, not the raw window pred)
        if gt_label == 1 and alarm_label == 1:
            outcome = "TP"
        elif gt_label == 0 and alarm_label == 1:
            outcome = "FP"
        elif gt_label == 1 and alarm_label == 0:
            outcome = "FN"
        else:
            outcome = "TN"
        run[outcome.lower()] += 1
        if global_run is not None:
            global_run[outcome.lower()] += 1

        records.append(
            {
                "clip_idx": clip_idx,
                "clip_start_frame": window[0],
                "clip_end_frame": window[1],
                "gt": gt_label,
                "pred": alarm_label,          # graded decision (queue vote)
                "window_pred": window_pred,    # raw per-window prediction (pre-vote)
                "queue_positive": queue_pos,   # # positive windows currently in the queue
                "queue_len": len(pred_queue),  # windows currently in the queue (<= queue_size)
                "pred_raw": pred_raw,
                "parse_failed": pred is None,
                "overlap_ratio": round(match_ratio, 4) if match_ratio is not None else None,
                "matched_event": list(match_ev) if match_ev else None,
                "gt_source": "unlabeled_assumed_positive" if force_positive else "label_overlap",
                "outcome": outcome,
                "inference_sec": round(timing["inference"], 4),
            }
        )

        # Overlay the alarm on this clip's frames (queue-vote positive only) and stream them out.
        if viz_path:
            alarm = alarm_label == 1
            desc = extract_description(pred_raw) if alarm else None
            for _, pil in buf:
                frame_img = draw_alarm(pil, alarm, label=pos_name, description=desc)
                if viz_writer is None:
                    viz_writer = VideoFromPIL(viz_path, fps=viz_fps)
                viz_writer.add_frame(frame_img)
            had_alarm = had_alarm or alarm

        buf.clear()

        if not args.quiet:
            window_name = (
                "unparsed" if pred is None else (pos_name if window_pred == 1 else "normal")
            )
            alarm_name = pos_name if alarm_label == 1 else "normal"
            gt_name = pos_name if gt_label == 1 else "normal"
            if force_positive:
                gt_detail = f"  [no label JSON -> assumed {pos_name}-positive (--unlabeled-as-positive)]"
            elif match_ev is not None:
                gt_detail = (
                    f"  [best match: event frames {match_ev[0]}-{match_ev[1]}, "
                    f"ratio={match_ratio:.3f} ({args.overlap_mode}) thr={args.overlap_threshold}]"
                )
            elif events:
                gt_detail = f"  [no overlap with {len(events)} {pos_name} event(s)]"
            else:
                gt_detail = f"  [no {pos_name} events in this video]"

            rp = run["tp"] / (run["tp"] + run["fp"]) if (run["tp"] + run["fp"]) > 0 else 0.0
            rr = run["tp"] / (run["tp"] + run["fn"]) if (run["tp"] + run["fn"]) > 0 else 0.0
            rf1 = 2 * rp * rr / (rp + rr) if (rp + rr) > 0 else 0.0
            mark = "OK" if gt_label == alarm_label else "XX"

            # corpus-wide running metrics (모든 영상 누적, 현재 기준)
            if global_run is not None:
                gtp, gfp, gfn = global_run["tp"], global_run["fp"], global_run["fn"]
                gp = gtp / (gtp + gfp) if (gtp + gfp) > 0 else 0.0
                gr = gtp / (gtp + gfn) if (gtp + gfn) > 0 else 0.0
                gf1 = 2 * gp * gr / (gp + gr) if (gp + gr) > 0 else 0.0
                g_clips = sum(global_run.values())

            # 클립별 VQA 추론 로그 (01_TuningFree_v2.py 콘솔 로그 스타일)
            print("==============================")
            print(f"File                : {os.path.basename(video_path)}")
            print(f"Event type          : {pos_name}")
            print(f"Clip                : {clip_idx}/~{est_total_clips}  (frames {window[0]}-{window[1]} / {total_frames})")
            print(f"FPS / sample step   : {fps:.1f} / {step}  (interval={args.interval}s, buffer={args.buffer_size})")
            print(f"Pre-processing time : {timing['preprocess']:.3f} sec")
            print(f"Inference time      : {timing['inference']:.3f} sec")
            print(f"Total time          : {timing['total']:.3f} sec")
            print(f"Model output (raw)  : {pred_raw!r}")
            print(f"Window prediction   : {window_name} ({window_pred})")
            print(f"Alarm queue         : {queue_pos}/{len(pred_queue)} positive  "
                  f"(fire if >= {args.queue_k} of last {args.queue_size})")
            print(f"Alarm (prediction)  : {alarm_name} ({alarm_label})")
            print(f"Ground truth        : {gt_name} ({gt_label}){gt_detail}")
            print(f"Result              : {mark}  ({outcome})")
            print(f"Running (this video): clips={clip_idx} TP={run['tp']} FP={run['fp']} FN={run['fn']} TN={run['tn']}  "
                  f"P={rp:.3f} R={rr:.3f} F1={rf1:.3f}")
            if global_run is not None:
                print(f"Running (ALL videos): clips={g_clips} TP={gtp} FP={gfp} FN={gfn} TN={global_run['tn']}  "
                      f"P={gp:.3f} R={gr:.3f} F1={gf1:.3f}")
            print("==============================")

        if args.max_clips_per_video and len(records) >= args.max_clips_per_video:
            break

    dropped = len(buf)
    cap.release()

    # Finalize the visualization video (drop it if --viz-only-alarms and nothing fired).
    if viz_writer is not None:
        viz_writer.save()
        if args.viz_only_alarms and not had_alarm:
            os.remove(viz_path)
            viz_path = None
        elif not args.quiet:
            print(f"[VIZ] saved -> {viz_path}")
    else:
        viz_path = None

    y_true = [r["gt"] for r in records]
    y_pred = [r["pred"] for r in records]
    metrics = binary_metrics(y_true, y_pred)
    metrics.update(
        {
            "video": os.path.basename(video_path),
            "fps": round(float(fps), 3),
            "total_frames": total_frames,
            "sample_step_frames": step,
            "n_target_events": len(events),
            "has_label": has_label,
            "unlabeled_assumed_positive": force_positive,
            "parse_failures": parse_fail,
            "dropped_tail_frames": dropped,
            "viz_path": viz_path,
            "avg_preprocess_sec": round(float(np.mean(pre_times)), 4) if pre_times else 0.0,
            "avg_inference_sec": round(float(np.mean(inf_times)), 4) if inf_times else 0.0,
            "avg_total_sec": round(float(np.mean(tot_times)), 4) if tot_times else 0.0,
        }
    )
    return records, metrics


def discover_videos(data_dir: str, pattern: str):
    """Yield (video_path, json_path) pairs. json_path may not exist (-> no events)."""
    videos = sorted(glob.glob(os.path.join(data_dir, pattern)))
    pairs = []
    for v in videos:
        j = os.path.splitext(v)[0] + ".json"
        pairs.append((v, j))
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-d", "--data-dir", required=True, help="Folder with *.mp4 and matching *.json files")
    parser.add_argument("-e", "--event-type", required=True, choices=sorted(EVENT_SPECS.keys()),
                        help="Event type to detect AND grade against this run")
    parser.add_argument("-m", "--model-path", default=DEFAULT_MODEL_PATH, help="InternVL3-2B path or HF id")
    parser.add_argument("-o", "--out-dir", default=os.path.join(os.path.dirname(__file__), "results"))
    parser.add_argument("--video-glob", default="*.mp4", help="Glob for videos inside data-dir")

    # Pipeline knobs (defaults match the fixed spec).
    parser.add_argument("--buffer-size", type=int, default=BUFFER_SIZE,
                        help="Frames per clip (== NUM_SEGMENTS). Set to 1 for single-image inference "
                             "(dynamic tiling) instead of multi-frame video inference.")
    parser.add_argument("--interval", type=float, default=INTERVAL_SEC, help="Sampling interval in seconds")
    parser.add_argument("--image-max-num", type=int, default=IMAGE_TILE_MAX_NUM,
                        help=f"With --buffer-size 1 (image inference), max dynamic tiles for the single "
                             f"image (InternVL3 image default {IMAGE_TILE_MAX_NUM}). Ignored for video.")

    # Alarm voting over a sliding queue of recent window predictions.
    parser.add_argument("--queue-size", type=int, default=QUEUE_SIZE,
                        help=f"Alarm voting window: number of most-recent per-window predictions kept "
                             f"in the sliding queue (default {QUEUE_SIZE}). 1 = no smoothing.")
    parser.add_argument("--queue-k", type=int, default=QUEUE_K,
                        help=f"Alarm fires when >= queue_k of the last --queue-size windows predicted "
                             f"the event (must satisfy 1 <= queue_k <= queue_size; default {QUEUE_K}).")

    # Grading knobs.
    parser.add_argument("--overlap-mode", default="min", choices=["min", "clip", "event", "iou"],
                        help="Overlap-ratio definition for GT positives (default: min = overlap coefficient)")
    parser.add_argument("--overlap-threshold", type=float, default=0.5,
                        help="Min overlap ratio for a clip to count as a GT positive (default: 0.5)")
    parser.add_argument("--unparsed-positive", action="store_true",
                        help="Treat unparseable model answers as positive (default: negative)")
    parser.add_argument("--unlabeled-as-positive", action="store_true",
                        help="Treat videos that have NO label JSON as all-positive for --event-type: "
                             "every clip is graded as a ground-truth positive (so a correct detection "
                             "counts as TP, a miss as FN). Use for folders curated to contain only the "
                             "target event, e.g. clips/violence/ with no frame-level labels. Videos that "
                             "DO have a label JSON are unaffected (graded by overlap as usual).")

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit-videos", type=int, default=0, help="Process at most N videos (0 = all; for debugging)")
    parser.add_argument("--max-clips-per-video", type=int, default=0,
                        help="Stop a video after N clips (0 = no limit; for debugging)")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-clip logging")

    # Visualization (off by default; requires ffmpeg on PATH).
    parser.add_argument("--save-viz", action="store_true",
                        help="Save an annotated .mp4 per video: the sampled frames, with a red "
                             "alert border + event label overlaid on clips predicted positive.")
    parser.add_argument("--viz-only-alarms", action="store_true",
                        help="With --save-viz, keep only the annotated videos that contain >=1 alarm.")
    parser.add_argument("--viz-fps", type=float, default=0.0,
                        help="Playback fps for the annotated video (0 = auto = 1/interval = real time).")
    args = parser.parse_args()

    if not (1 <= args.queue_k <= args.queue_size):
        raise SystemExit(
            f"--queue-k ({args.queue_k}) must satisfy 1 <= queue_k <= --queue-size ({args.queue_size})"
        )

    if args.buffer_size != NUM_SEGMENTS:
        print(f"[INFO] buffer_size={args.buffer_size} (spec default NUM_SEGMENTS={NUM_SEGMENTS}); "
              f"the model receives {args.buffer_size} frames/clip and the <image> prefix matches automatically.")

    spec = EVENT_SPECS[args.event_type]
    pairs = discover_videos(args.data_dir, args.video_glob)
    if not pairs:
        raise SystemExit(f"No videos matched {args.video_glob} in {args.data_dir}")

    # Process GT-positive videos first. A video is target-positive when its label JSON holds
    # >=1 interval of that class — OR, under --unlabeled-as-positive, when it has no label JSON
    # at all. Sort is stable on the secondary key, so ordering stays alphabetical within each
    # group (positive first, then the rest).
    def _is_target_video(json_path: str) -> bool:
        if not os.path.isfile(json_path):
            return args.unlabeled_as_positive
        return len(load_target_events(json_path, spec["class_names"])) > 0

    is_target = {v: _is_target_video(j) for v, j in pairs}
    pairs.sort(key=lambda vj: (0 if is_target[vj[0]] else 1, vj[0]))
    if args.limit_videos > 0:
        pairs = pairs[: args.limit_videos]
    n_with = sum(1 for v, _ in pairs if is_target[v])

    mode_note = " | unlabeled-as-positive=ON" if args.unlabeled_as_positive else ""
    print(f"[INFO] event-type={args.event_type} | videos={len(pairs)} "
          f"({n_with} graded as {args.event_type}-positive -> processed first) | "
          f"interval={args.interval}s | buffer={args.buffer_size} | "
          f"alarm>={args.queue_k}/{args.queue_size} windows | "
          f"overlap={args.overlap_mode}>={args.overlap_threshold}{mode_note}")

    if args.buffer_size == 1:
        print(f"[INFO] inference mode = IMAGE (single frame, dynamic tiling up to "
              f"image_max_num={args.image_max_num} tiles + thumbnail, one <image>)")
    else:
        print(f"[INFO] inference mode = VIDEO ({args.buffer_size} frames/clip, 1 tile each, "
              f"Frame-prefixed <image> per frame)")

    streamer = InternVL3Streamer(args.model_path, device=args.device, image_max_num=args.image_max_num)

    os.makedirs(args.out_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H-%M-%S")

    viz_dir = None
    if args.save_viz:
        viz_dir = os.path.join(args.out_dir, f"{timestamp}_{args.event_type}_viz")
        os.makedirs(viz_dir, exist_ok=True)
        only = "  (only videos with >=1 alarm)" if args.viz_only_alarms else ""
        print(f"[INFO] alarm visualization ON -> {viz_dir}{only}")

    per_video = []
    all_true, all_pred = [], []
    all_records = {}
    global_run = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}  # shared across videos -> corpus-wide running F1
    t_start = time.time()

    for i, (video_path, json_path) in enumerate(pairs, 1):
        name = os.path.basename(video_path)
        print(f"\n[{i}/{len(pairs)}] {name}")
        records, metrics = process_video(streamer, video_path, json_path, spec, args, global_run, viz_dir=viz_dir)
        if metrics is None:
            continue
        all_records[name] = records
        all_true.extend(r["gt"] for r in records)
        all_pred.extend(r["pred"] for r in records)
        per_video.append(metrics)
        print(
            f"  -> clips={metrics['n_clips']} events={metrics['n_target_events']} "
            f"TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']} TN={metrics['tn']} "
            f"P={metrics['precision']:.3f} R={metrics['recall']:.3f} F1={metrics['f1']:.3f} "
            f"avg_inf={metrics['avg_inference_sec']:.3f}s"
        )

    final = binary_metrics(all_true, all_pred)
    elapsed = time.time() - t_start
    tot_clips = sum(m["n_clips"] for m in per_video)
    avg_inf = (
        sum(m["avg_inference_sec"] * m["n_clips"] for m in per_video) / tot_clips
        if tot_clips > 0 else 0.0
    )
    final["avg_inference_sec"] = round(avg_inf, 4)

    # ---- Console summary -------------------------------------------------
    print("\n" + "=" * 78)
    print(f"FINAL  (event-type={args.event_type}, alarm>={args.queue_k}/{args.queue_size} windows, "
          f"overlap {args.overlap_mode}>={args.overlap_threshold})")
    print("=" * 78)
    print(f"  videos        : {len(per_video)}")
    print(f"  total clips   : {final['n_clips']}")
    print(f"  TP/FP/FN/TN   : {final['tp']}/{final['fp']}/{final['fn']}/{final['tn']}")
    print(f"  precision     : {final['precision']:.4f}")
    print(f"  recall        : {final['recall']:.4f}")
    print(f"  F1-score      : {final['f1']:.4f}")
    print(f"  accuracy      : {final['accuracy']:.4f}")
    print(f"  avg inference : {avg_inf:.3f} sec/clip")
    print(f"  elapsed       : {elapsed:.1f}s")
    print("=" * 78)

    # ---- Persist results -------------------------------------------------
    out = {
        "event_type": args.event_type,
        "data_dir": os.path.abspath(args.data_dir),
        "model_path": args.model_path,
        "config": {
            "num_segments": NUM_SEGMENTS,
            "buffer_size": args.buffer_size,
            "interval_sec": args.interval,
            "inference_mode": "image" if args.buffer_size == 1 else "video",
            "max_num_tiles": args.image_max_num if args.buffer_size == 1 else FRAME_PER_TILE_MAX_NUM,
            "image_max_num": args.image_max_num,
            "max_new_tokens": MAX_NEW_TOKEN,
            "input_size": INPUT_SIZE,
            "dtype": str(MODEL_INF_DATA_TYPE),
            "queue_size": args.queue_size,
            "queue_k": args.queue_k,
            "overlap_mode": args.overlap_mode,
            "overlap_threshold": args.overlap_threshold,
            "unparsed_positive": args.unparsed_positive,
            "unlabeled_as_positive": args.unlabeled_as_positive,
        },
        "final_metrics": final,
        "per_video_metrics": per_video,
        "elapsed_sec": elapsed,
    }
    summary_path = os.path.join(args.out_dir, f"{timestamp}_{args.event_type}_streaming_eval.json")
    with open(summary_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    detail_path = os.path.join(args.out_dir, f"{timestamp}_{args.event_type}_clip_records.json")
    with open(detail_path, "w") as f:
        json.dump(all_records, f, indent=2, ensure_ascii=False)

    print(f"[INFO] summary -> {summary_path}")
    print(f"[INFO] per-clip records -> {detail_path}")


if __name__ == "__main__":
    main()
