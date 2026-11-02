#!/usr/bin/env zsh
# Major-revision empirical programme.
#
# Everything runs on one MPS device, so the stages are strictly sequential
# and each is --resume-able: an interrupt continues rather than restarts.
#
# IMPORTANT: the evaluator's results.json carries only the methods of the run
# that wrote it -- a per-method invocation therefore REPLACES the tree rather
# than adding to it. Every stage below that accumulates methods either scores
# them in one invocation or writes each into its own directory and merges with
# experiments/merge_tree.py. Writing per-method runs into one shared directory
# silently destroys the previous method's column.
#
# Stage order is by reviewer value, not by section order:
#   1. The two new baseline adapters on the UIEB test split (fast).
#   2. Ablations re-run at n=100 (up from 40), now over nine loss groups.
#   3. Two extra PhysINR-Zero meta-training seeds, for mean +- sd and a CI.
#   4. CycleGAN at a reduced budget (see the note in the stage).
#   5. PhysINR-Zero re-run on all five datasets against the current code.
#   6. Full-890 UIEB evaluation of every method.
#   7. YOLOv8n fine-tuned on DUO (~10 h; long, so late).
#   8. Frozen-detector mAP pass on DUO, using the stage-7 weights.
#   9. Merge every per-method tree into the trees the table builder reads.
#
# Measured on this machine (M1, MPS): PhysINR-Zero ~8 s/image at stride 2,
# Phaseformer ~7 s/image, Zero-DCE ~1.7 s/image. CycleGAN trains at ~4.3 s/it,
# so the published 200-epoch protocol is ~150 h and is *not* run; a reduced
# budget is used instead and reported as such.
set -u
cd "$(dirname "$0")/.."

export HYDROBENCH_DATA=$PWD/datasets
export HYDROBENCH_MODELS=$PWD/weights
export HYDROBENCH_THIRD_PARTY=$PWD/third_party

PY=python3
RES=hydro_bench_results           # the baseline spine (one tree per dataset)
NEW=experiments/new_baselines     # Zero-DCE + Phaseformer (UIEB only)
ABL=experiments/physinr_ablations_n100
SEEDS=experiments/physinr_seeds
CYC_EVAL=experiments/cyclegan_eval
PHYSINR=experiments/physinr_results
FULL=experiments/full890          # one subdirectory per method
DUO_DIR=experiments/duo_results   # one subdirectory per method
BASEV2=experiments/baselines_v2   # baselines + new methods, every dataset
MERGED=experiments/merged_results # what the table/figure builders read

# zsh does not field-split unquoted expansions, so these are arrays; expand
# them as "$FR[@]" at the call sites, never bare $FR.
FR=(psnr ssim ms_ssim lpips_alex lpips_vgg)
NR=(uciqe uiqm uicm uism uiconm ccf)
DATASETS=(UIEB LSUI EUVP RUIE UCCS)
ABL_DATASET=UIEB
ABL_LIMIT=100

# PhysINR-Zero was refactored during this revision (colour-cast handling). The
# numbers in the manuscript predate the refactor, so every PhysINR-Zero column
# is regenerated against the refactored code and the checkpoint it trains. The
# checkpoint is written by a separate meta-training run; wait for it rather
# than training a duplicate on the same device.
CAST_CKPT=weights/physinr_zero_cast.pt

log () { echo "[ $(date '+%m-%d %H:%M:%S') ] $*"; }

# ------------------------------------------------------------------ queue
# Never start while another GPU job is in flight; MPS contention degrades
# both jobs by an order of magnitude. The patterns are specific enough not to
# match this script's own command line.
while pgrep -f "cyclegan/train.py" > /dev/null \
   || pgrep -f "ultralytics.*model.train" > /dev/null \
   || pgrep -f "hydro_bench.methods.deep.physinr_zero.cli meta-train" > /dev/null \
   || pgrep -f "hydro_bench.methods.deep.physinr_zero.cli compare" > /dev/null \
   || pgrep -f "hydro_bench/run_benchmark.py" > /dev/null; do
  log "waiting for a prior GPU job to finish..."
  sleep 120
done

# The refactored method needs the checkpoint the refactor's meta-training
# produces; without it the stages below would score an old checkpoint against
# new physics and report a configuration that was never trained.
while [ ! -f "$CAST_CKPT" ]; do
  log "waiting for the refactored checkpoint ($CAST_CKPT)..."
  sleep 120
done
log "checkpoint present; pointing the method config at it"

# One place defines the default checkpoint the adapter resolves, so the
# ablations and the all-method stages all see the same weights.
if grep -q "checkpoint: physinr_zero_meta.pt}" hydro_bench/configs/methods.yaml; then
  sed -i '' 's|params: {checkpoint: physinr_zero_meta.pt}|params: {checkpoint: physinr_zero_cast.pt}|' \
      hydro_bench/configs/methods.yaml
  log "methods.yaml: PhysINR-Zero default checkpoint -> physinr_zero_cast.pt"
else
  log "methods.yaml: already points at the refactored checkpoint"
fi

log "GPU free; starting the revision programme"

# A single invocation scores every requested method into one results.json, so
# methods that must coexist are requested together here.
paired_many () {  # out-dir dataset limit method... -- extra args
  out=$1; shift; ds=$1; shift; lim=$1; shift
  $PY hydro_bench/run_benchmark.py run --dataset "$ds" --split test \
      --limit "$lim" --methods "$@" --metrics "$FR[@]" "$NR[@]" \
      --out-dir "$out" --resume --quiet
}

# A per-method invocation replaces the tree, so each method gets its own
# directory and merge_tree.py folds them together afterwards. Use solo() when
# several methods run against one dataset and solo_tree() when a dataset tree
# holds a single method -- the latter keeps the tree at the path merge_results
# and merge_tree expect (<dir>/<DATASET>__<split>/results.json).
solo () {  # out-root dataset method -- extra args
  root=$1; shift; ds=$1; shift; meth=$1; shift
  $PY hydro_bench/run_benchmark.py run --dataset "$ds" --split test \
      --methods "$meth" "$@" --out-dir "$root/$meth" --resume --quiet
}
solo_tree () {  # out-dir dataset method -- extra args
  out=$1; shift; ds=$1; shift; meth=$1; shift
  $PY hydro_bench/run_benchmark.py run --dataset "$ds" --split test \
      --methods "$meth" "$@" --out-dir "$out" --resume --quiet
}

# ---------------------------------------------------------------- stage 1
log "=== [1/9] new baselines on the UIEB test split ==="
paired_many "$NEW/UIEB" UIEB 134 Zero-DCE Phaseformer \
  || log "new-baseline run failed on UIEB"
log "stage 1 complete"

# ---------------------------------------------------------------- stage 2
log "=== [2/9] ablations at n=$ABL_LIMIT (nine loss groups) ==="
for G in phys cycle color struct spatial jerlov prior cast bound; do
  solo "$ABL/no_$G" "$ABL_DATASET" PhysINR-Zero \
      --limit "$ABL_LIMIT" --metrics "$FR[@]" "$NR[@]" \
      --params "PhysINR-Zero:drop_group=$G" \
    || log "ablation no_$G failed"
done
for S in 1 3; do
  solo_tree "$ABL/stride_$S" "$ABL_DATASET" PhysINR-Zero \
      --limit "$ABL_LIMIT" --metrics "$FR[@]" "$NR[@]" \
      --params "PhysINR-Zero:loss_stride=$S" \
    || log "ablation stride_$S failed"
done
for N in 5 20; do
  solo_tree "$ABL/steps_$N" "$ABL_DATASET" PhysINR-Zero \
      --limit "$ABL_LIMIT" --metrics "$FR[@]" "$NR[@]" \
      --params "PhysINR-Zero:adapt_steps=$N" \
    || log "ablation steps_$N failed"
done
for T in 2.0 4.0; do
  solo_tree "$ABL/tau_max_$T" "$ABL_DATASET" PhysINR-Zero \
      --limit "$ABL_LIMIT" --metrics "$FR[@]" "$NR[@]" \
      --params "PhysINR-Zero:tau_max=$T" \
    || log "ablation tau_max_$T failed"
done
for D in 0.5 4.0; do
  solo_tree "$ABL/delta_max_$D" "$ABL_DATASET" PhysINR-Zero \
      --limit "$ABL_LIMIT" --metrics "$FR[@]" "$NR[@]" \
      --params "PhysINR-Zero:delta_max=$D" \
    || log "ablation delta_max_$D failed"
done
log "stage 2 complete"

# ---------------------------------------------------------------- stage 3
log "=== [3/9] two extra meta-training seeds ==="
# The published checkpoint is seed 1234. Seeds 7 and 2025 are meta-trained
# from scratch on the same clean-photo pool, so the reported mean +- sd is
# over independent initialisations rather than one network's adaptation.
for SEED in 7 2025; do
  PT=weights/physinr_zero_meta_seed${SEED}.pt
  PV=weights/physinr_zero_meta_seed${SEED}.provenance.json
  if [ -f "$PT" ] && [ -f "$PV" ] && \
     $PY -c "import json,sys; sys.exit(0 if json.load(open('$PV')).get('tasks_completed',0) > 0 else 1)"; then
    log "seed $SEED checkpoint present; skipping meta-training"
  else
    rm -f "$PT" "$PV"
    $PY -m hydro_bench.methods.deep.physinr_zero.cli meta-train --source datasets/clean_photos \
        --out "$PT" --seed "$SEED" \
      || { log "seed $SEED meta-train FAILED"; continue; }
  fi
  # load_weights() joins a relative path onto HYDROBENCH_MODELS, so the
  # override is the bare filename, not the repo-relative path.
  PT_NAME="physinr_zero_meta_seed${SEED}.pt"
  solo_tree "$SEEDS/seed_$SEED" UIEB PhysINR-Zero \
      --limit 134 --metrics "$FR[@]" "$NR[@]" \
      --params "PhysINR-Zero:checkpoint=$PT_NAME" \
    || log "seed $SEED evaluation failed"
done
log "stage 3 complete"

# ---------------------------------------------------------------- stage 4
log "=== [4/9] CycleGAN (reduced budget) ==="
# No pretrained underwater CycleGAN exists, so one is trained here. The
# published protocol is 100 + 100 epochs at ~4.3 s/iter on this hardware,
# i.e. ~150 h, which is not feasible alongside the rest of the revision.
# The reduced budget below is what this machine can afford; the manuscript
# states the budget explicitly so the row is not mistaken for a fully
# converged CycleGAN.
CYC_DIR=third_party/cyclegan
CYC_NAME=uieb_cyclegan_reduced
CYC_CKPT="$CYC_DIR/checkpoints/$CYC_NAME/latest_net_G.pth"
if [ ! -f "$CYC_CKPT" ]; then
  ( cd "$CYC_DIR" && PYTHONPATH="$PWD" $PY -u train.py \
      --dataroot ./datasets/uieb_train --name "$CYC_NAME" \
      --model cycle_gan --direction AtoB \
      --n_epochs 8 --n_epochs_decay 7 --epoch_count 1 \
      --save_epoch_freq 5 --save_latest_freq 1000 --print_freq 200 --no_html \
      --load_size 286 --crop_size 256 --num_threads 2 \
  ) >> experiments/logs_cyclegan_train.txt 2>&1 \
    || log "CycleGAN training FAILED"
fi
if [ -f "$CYC_CKPT" ]; then
  cp -f "$CYC_CKPT" weights/cyclegan_uieb_latest.pth
  # Re-stage the checkpoint directory the adapter's loader expects.
  mkdir -p weights/cyclegan_ckpt/uieb_cyclegan
  cp -f "$CYC_CKPT" weights/cyclegan_ckpt/uieb_cyclegan/latest_net_G.pth
  solo_tree "$CYC_EVAL/UIEB" UIEB CycleGAN \
      --limit 134 --metrics "$FR[@]" "$NR[@]" \
      --params "CycleGAN:checkpoint=cyclegan_ckpt/uieb_cyclegan/latest_net_G.pth" \
    || log "CycleGAN evaluation failed"
else
  log "WARNING: CycleGAN checkpoint missing; that baseline is not reported"
fi
log "stage 4 complete"

# ---------------------------------------------------------------- stage 5
log "=== [5/9] PhysINR-Zero re-run on all five datasets ==="
# The method was refactored during this revision; the columns in the
# manuscript predate it. This regenerates every PhysINR-Zero column against
# the current code and the refactored checkpoint, one dataset per invocation
# (each is its own tree, so nothing is clobbered).
for DS in $DATASETS; do
  case "$DS" in
    UIEB) LIM=134 ;; LSUI|EUVP) LIM=120 ;; RUIE) LIM=100 ;; UCCS) LIM=45 ;;
  esac
  # methods.yaml was repointed at the refactored checkpoint above; the override
  # is stated explicitly (bare filename, see stage 3) so a later config change
  # cannot silently re-couple these columns to a different checkpoint.
  solo_tree "$PHYSINR" "$DS" PhysINR-Zero \
      --limit "$LIM" --metrics "$FR[@]" "$NR[@]" \
      --params "PhysINR-Zero:checkpoint=physinr_zero_cast.pt" \
    || log "PhysINR-Zero re-run failed on $DS"
done
log "stage 5 complete"

# ---------------------------------------------------------------- stage 6
log "=== [6/9] full-890 UIEB ==="
$PY experiments/make_uieb_full890_manifest.py
METHODS_ALL="$($PY -c "
from hydro_bench.core.base_algorithm import registered_algorithms
print(' '.join(sorted(n for n in registered_algorithms() if n != 'ExampleOffline')))
")"
for M in $=METHODS_ALL; do
  solo "$FULL/UIEB-890" "UIEB-890" "$M" \
      --metrics "$FR[@]" "$NR[@]" \
    || log "method $M failed on UIEB-890"
done
log "stage 6 complete"

# ---------------------------------------------------------------- stage 7
# The detector is trained late because it is the longest single job
# (~10 h at ~11 min/epoch for 60 epochs) and nothing above needs its weights.
# Four epochs were already completed by an earlier run and are resumed from
# last.pt rather than discarded.
log "=== [7/9] YOLOv8n on DUO (resumed) ==="
YOLO_RUN=weights/duo_det/yolov8n_duo
YOLO_W="$YOLO_RUN/weights/best.pt"
YOLO_LAST="$YOLO_RUN/weights/last.pt"
if [ -f "$YOLO_W" ] && [ -f "$YOLO_RUN/results.csv" ] && \
   [ "$(tail -1 "$YOLO_RUN/results.csv" | cut -d, -f1)" -ge 59 ]; then
  log "YOLOv8n already trained to 60 epochs; skipping"
elif [ -f "$YOLO_LAST" ]; then
  log "resuming YOLOv8n from $(tail -1 "$YOLO_RUN/results.csv" | cut -d, -f1) completed epochs"
  $PY -u -c "
from ultralytics import YOLO
YOLO('$YOLO_LAST').train(resume=True)
" >> experiments/logs_yolo_train.txt 2>&1 \
    || log "YOLOv8n resume FAILED"
else
  log "training YOLOv8n on the DUO train split (1360 images) from scratch"
  $PY -u -c "
from ultralytics import YOLO
YOLO('yolov8n.pt').train(
    data='weights/duo_data.yaml',
    epochs=60, imgsz=640, batch=8, device='mps',
    project='weights/duo_det', name='yolov8n_duo',
    workers=2, patience=25, val=True, plots=False, exist_ok=True,
)
" >> experiments/logs_yolo_train.txt 2>&1 \
    || log "YOLOv8n training FAILED"
fi
log "stage 7 complete"

# ---------------------------------------------------------------- stage 8
log "=== [8/9] frozen-detector mAP on DUO ==="
if [ -f "$YOLO_W" ]; then
  cp -f "$YOLO_W" weights/yolov8n_duo.pt
  METHODS_ALL="$($PY -c "
from hydro_bench.core.base_algorithm import registered_algorithms
print(' '.join(sorted(n for n in registered_algorithms() if n != 'ExampleOffline')))
")"
  for M in $=METHODS_ALL; do
    case "$M" in
      PhysINR-Zero) EXTRA="--params PhysINR-Zero:checkpoint=$CAST_CKPT" ;;
      *)            EXTRA="" ;;
    esac
    # shellcheck disable=SC2086
    $PY hydro_bench/run_benchmark.py run --dataset DUO --split test \
        --methods "$M" --metrics "$NR[@]" \
        --out-dir "$DUO_DIR/$M" --resume --quiet --save-images \
        --detection --detection-weights weights/yolov8n_duo.pt \
        --detection-label-dir datasets/DUO/test/labels \
        --detection-imgsz 640 $EXTRA \
      || log "DUO detection failed for $M"
  done
else
  log "WARNING: YOLOv8n weights missing; the DUO detection pass is skipped"
fi
log "stage 8 complete"

# ---------------------------------------------------------------- stage 9
log "=== [9/9] merge the per-method trees ==="
# Baselines + the new methods, for every dataset: the new baselines and
# CycleGAN only ran on UIEB, and merge_tree skips a missing input, so the
# other datasets keep their existing baseline columns.
for DS in $DATASETS; do
  $PY experiments/merge_tree.py --dataset "$DS" \
      --inputs "$RES" "$NEW/UIEB" "$CYC_EVAL/UIEB" \
      --out "$BASEV2" \
    || log "baseline merge failed for $DS"
done
# merge_results renames the ablation columns (the variant directories all
# store their scores under the bare method name) and folds in PhysINR-Zero.
$PY experiments/merge_results.py --baseline-dir "$BASEV2" \
    --physinr-dir "$PHYSINR" --ablation-dir "$ABL" \
    --out-dir "$MERGED" \
  || log "merged-results build failed"
$PY experiments/merge_tree.py --dataset UIEB-890 \
    --inputs $FULL/UIEB-890/* --out experiments/merged_results_full890 \
  || log "full-890 merge failed"
if [ -d "$DUO_DIR" ]; then
  $PY experiments/merge_tree.py --dataset DUO \
      --inputs $DUO_DIR/* --out experiments/merged_results_duo \
    || log "DUO merge failed"
fi
log "stage 9 complete"

log "=== REVISION PROGRAMME COMPLETE ==="
