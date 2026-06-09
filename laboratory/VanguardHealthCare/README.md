# VanguardHealthCare — Streaming InternVL3 Evaluation

Streaming (deque-based) evaluation of **InternVL3-2B** on the *VanguardHealthCare 2ndPoC*
CCTV dataset. For each clip the model makes a binary call — *event present* vs *normal* —
for a single event type (`violence` or `climbing`), and the harness grades those calls
against frame-level ground-truth intervals to report **precision / recall / F1**.

## Quick start

From the **`train_eval_toolkit` repo root**, set `DATA_DIR` in the launcher once, then:

```bash
bash laboratory/VanguardHealthCare/01_StreamingEval_InternVL3.sh            # violence (default)
bash laboratory/VanguardHealthCare/01_StreamingEval_InternVL3.sh climbing   # climbing
```

The script resolves its own location, so it runs the same from any working directory.

<a id="sampling-window"></a>

> ### ⚠️ Default sampling: `--interval 0.09` + `--buffer-size 12` (≈ 1-second clip)
>
> The launcher's defaults are **`--interval 0.09` and `--buffer-size 12`** — and this is
> deliberate, **not** the script's generic spec defaults (`interval 1.0s`). The fine-tuned
> checkpoint
> [`InternVL3-2B_gangnam_rwf2000_gj_cctv_scvdALL_NOweapon_no_split`](#model-checkpoint)
> was **trained on 12 frames sampled at a 0.09 s interval**, i.e. **one ≈ 1-second clip per
> inference** (12 × 0.09 s ≈ 1.08 s).
>
> **Why ~1 second:** violence is a very short-lived event — it appears and is over within a
> fraction of a second to a second. So the model is fed only ~1 second of footage per
> prediction, matching how it was trained. **Keep `--interval` and `--buffer-size` in sync
> with the model's training window**; changing them (e.g. to the 1.0 s spec default) feeds
> the model a temporal context it was never trained on and degrades accuracy.

## Files

| File | Purpose |
|---|---|
| `01_StreamingEval_InternVL3.py` | Main evaluation script (streaming inference + grading + metrics). |
| `01_StreamingEval_InternVL3.sh` | Convenience launcher — calls the `.py` next to it with the `internvl` conda env. |
| `prompts.yaml` | Per-event-type prompts. Edit prompt text here — no code change needed. |
| `results/` | Auto-created output (summary + per-clip records JSON). |

## How it works

1. **Discover** — walks `--data-dir` for `*.mp4` clips, each paired with a matching
   `*.json` label file (same basename).
2. **Stream** — reads frames sequentially with OpenCV, sampling one frame every
   `--interval` seconds into a deque. When the deque reaches `--buffer-size` frames, it
   runs **one** InternVL3 inference over that window, then clears the deque
   (non-overlapping sliding window).
3. **Predict** — each inference returns a binary label for the chosen `--event-type`.
   The model output (a small JSON object per `prompts.yaml`) is parsed robustly — bare
   keyword, a `category` field, or truncated JSON all work (15-token budget).
4. **Grade** — a clip's frame window is a ground-truth positive when it overlaps a
   target-class event interval by at least `--overlap-threshold` under the chosen
   `--overlap-mode` (overlap-coefficient by default).
5. **Report** — per-video and corpus-wide precision / recall / F1 / accuracy, printed
   live per clip and saved to `results/`.

### Scoring scope

Inference runs on **every** video in the folder, but grading is done for **only** the
single `--event-type` passed on the command line. Videos with no events — or with events
of a *different* class — contribute only negatives, so a false firing on them counts as a
false positive. Event-bearing videos are processed first (alphabetical within each group).

<a id="unlabeled-folders"></a>

### Unlabeled folders → `--unlabeled-as-positive`

Some folders are curated to contain **only** the target event but ship **without** per-clip
label JSONs — e.g. `.../clips/violence/` with bare `*.mp4`. By default such a clip has no
events, so it's graded as a negative and a *correct* violence detection is wrongly scored as
a **false positive**.

Pass `--unlabeled-as-positive` to treat every video that has **no label JSON** as
all-positive for `--event-type`: each clip becomes a ground-truth positive, so a correct
detection counts as **TP** and a miss as **FN**. Videos that *do* have a label JSON are
unaffected — still graded by overlap. This is the launcher's default for the `clips/violence`
folder.

## Expected data format

```
data-dir/
├── clip_0001.mp4
├── clip_0001.json
├── clip_0002.mp4
└── clip_0002.json
```

Each `*.json` label holds parallel event maps (the `.json` may be absent → the clip has no
events):

```json
{
  "annotations_frame": { "S1": 120, "E1": 245, "S2": 600, "E2": 712 },
  "annotations_class": { "E1": "violence", "E2": "climbing" }
}
```

`S{n}`/`E{n}` are start/end **frame indices**; `annotations_class["E{n}"]` is that event's
class. Only intervals whose class matches the run's `--event-type` are graded as positives.

## Usage

### Via the launcher (recommended)

Set `DATA_DIR` in `01_StreamingEval_InternVL3.sh`, then:

```bash
./01_StreamingEval_InternVL3.sh            # event-type defaults to "violence"
./01_StreamingEval_InternVL3.sh climbing   # override event-type
```

The launcher pins the `internvl` conda env
(`/workspace/sangrak/anaconda3/envs/internvl/bin/python`), which carries the
transformers / decord / cv2 stack InternVL3 needs, and runs on `CUDA_VISIBLE_DEVICES=0`.

### Direct invocation

```bash
python 01_StreamingEval_InternVL3.py \
    --data-dir /path/to/clips \
    --event-type violence \
    --model-path /path/to/InternVL3-2B \
    --buffer-size 12 \
    --interval 1.0 \
    --overlap-mode min \
    --overlap-threshold 0.5
```

### Key arguments

| Arg | Default | Meaning |
|---|---|---|
| `-d, --data-dir` | *(required)* | Folder of `*.mp4` + matching `*.json`. |
| `-e, --event-type` | *(required)* | `violence` or `climbing` — detected **and** graded this run. |
| `-m, --model-path` | `../../ckpts/InternVL3-2B` | InternVL3-2B local path or HF id. |
| `--buffer-size` | `12` | Frames per clip / inference (= `NUM_SEGMENTS`). |
| `--interval` | `1.0` | Sampling interval in seconds. |
| `--overlap-mode` | `min` | `min` (overlap coeff.) · `clip` · `event` · `iou`. |
| `--overlap-threshold` | `0.5` | Min overlap ratio for a clip to be a GT positive. |
| `--unparsed-positive` | off | Treat unparseable answers as positive (default: negative). |
| `--unlabeled-as-positive` | off | Grade videos that have **no label JSON** as all-positive for `--event-type` (correct detection → TP, miss → FN). For folders curated to hold only the target event. See [below](#unlabeled-folders). |
| `--limit-videos`, `--max-clips-per-video` | `0` (all) | Debug caps. |
| `--quiet` | off | Suppress per-clip logging. |

> **Note — launcher vs. spec defaults:** `01_StreamingEval_InternVL3.sh` deliberately runs
> at **`--interval 0.09 --buffer-size 12`** (one ≈ 1-second clip per inference) to match the
> fine-tuned checkpoint's training window — see the
> [callout above](#sampling-window).
> The `--interval 1.0` shown in *Direct invocation* is the script's generic 2ndPoC spec
> default; the launcher overrides it on purpose. Other fixed knobs (`max_num=1`,
> `max_new_tokens=15`, `bfloat16`, 448×448) follow the spec. The `<image>` prefix is built
> from the actual buffered frame count, so any `buffer-size` is technically valid — but use
> 12 with this checkpoint.

## Model checkpoint

The launcher uses a fine-tuned checkpoint hosted on Hugging Face:

> **PIA-SPACE-LAB/InternVL3-2B_gangnam_rwf2000_gj_cctv_scvdALL_NOweapon_no_split**
> <https://huggingface.co/PIA-SPACE-LAB/InternVL3-2B_gangnam_rwf2000_gj_cctv_scvdALL_NOweapon_no_split/tree/main>

**Training window:** this checkpoint was trained on **12 frames sampled every 0.09 s
(≈ 1-second clips)**. Run it with `--interval 0.09 --buffer-size 12` (the launcher's
defaults) so inference matches training — see the
[sampling callout](#sampling-window).

Download it into `ckpts/` at the repo root (the path the `.sh` expects):

```bash
huggingface-cli download \
    PIA-SPACE-LAB/InternVL3-2B_gangnam_rwf2000_gj_cctv_scvdALL_NOweapon_no_split \
    --local-dir ckpts/InternVL3-2B_gangnam_rwf2000_gj_cctv_scvdALL_NOweapon_no_split
```

Or pass any other InternVL3-2B checkpoint (local dir or HF id) via `--model-path`.

## Output

Written to `--out-dir` (default `./results/`), timestamped:

- `{ts}_{event}_streaming_eval.json` — run config + per-video and final metrics.
- `{ts}_{event}_clip_records.json` — every clip's window, prediction, raw model output,
  matched event, overlap ratio, and outcome (TP/FP/FN/TN).

## Requirements

- `internvl` conda env (transformers, torch, torchvision, opencv-python, pyyaml, numpy, Pillow).
- A single CUDA GPU.
- An InternVL3-2B checkpoint — see [Model checkpoint](#model-checkpoint).
