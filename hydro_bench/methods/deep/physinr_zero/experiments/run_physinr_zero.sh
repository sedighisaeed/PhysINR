#!/usr/bin/env bash
# PhysINR-Zero experiments: meta-training, the main comparison, and the
# ablations of the objective and the physical bounds.
#
# The runner OVERWRITES results.json at the end of a completed run (its
# partial.json checkpoint is deleted on success), so every variant here writes
# to its own output directory. The baseline scores in hydro_bench_results/ are
# never touched; experiments/merge_results.py combines them afterwards.
#
# Budget (measured on this machine, MPS): meta-training ~55 min, the main
# comparison ~70 min, ablations ~90 min. Everything is --resume-able, so an
# interrupt continues rather than restarting.
set -u
cd "$(dirname "$0")/.."

export HYDROBENCH_DATA=$PWD/datasets
export HYDROBENCH_MODELS=$PWD/weights
export HYDROBENCH_THIRD_PARTY=$PWD/third_party

PY=python3
OUT=experiments/physinr_results
ABL=experiments/physinr_ablations
SRC=datasets/clean_photos
METHOD="PhysINR-Zero"
FR="psnr ssim ms_ssim lpips_alex lpips_vgg"
NR="uciqe uiqm uicm uism uiconm ccf"

# Ablation subset: UIEB has reference images, so FR metrics are available,
# and 40 images keeps each variant affordable at ~8 s/image.
ABL_DATASET=UIEB
ABL_LIMIT=40

paired ()   {  # out-dir dataset limit [extra args...]
  $PY hydro_bench/run_benchmark.py run --dataset "$2" --split test \
      --limit "$3" --methods "$METHOD" --metrics $FR $NR \
      --out-dir "$1" --resume --quiet "${@:4}"
}
unpaired () {  # out-dir dataset limit [extra args...]
  $PY hydro_bench/run_benchmark.py run --dataset "$2" --split test \
      --limit "$3" --methods "$METHOD" --metrics $NR \
      --out-dir "$1" --resume --quiet "${@:4}"
}

# ---------------------------------------------------------------- stage 1
echo "=== [1/4] meta-training on the clean photo pool ==="
# The guard is stronger than a file check: a *placeholder* checkpoint written
# for plumbing tests also exists as weights/physinr_zero_meta.pt, and
# benchmarking it would report the scores of a random init (demonstrated to
# diverge). Only a provenance record showing completed tasks counts as real.
META_PT=weights/physinr_zero_meta.pt
META_PROV=weights/physinr_zero_meta.provenance.json
if [ -f "$META_PT" ] && [ -f "$META_PROV" ] && \
   $PY -c "import json,sys; sys.exit(0 if json.load(open('$META_PROV')).get('tasks_completed',0) > 0 else 1)"; then
  echo "meta-trained checkpoint already present; skipping (delete $META_PT to redo)"
else
  rm -f "$META_PT" "$META_PROV"
  $PY -m hydro_bench.methods.deep.physinr_zero.cli meta-train --source "$SRC" \
      --out "$META_PT"
fi

# ---------------------------------------------------------------- stage 2
echo "=== [2/4] main comparison ==="
# Same image limits as run_all.sh so the rows are comparable with the baselines.
unpaired "$OUT" UCCS 45
unpaired "$OUT" RUIE 100
paired   "$OUT" EUVP 120
paired   "$OUT" LSUI 120
# UIEB is the full reference set the ablations are run on.
paired   "$OUT" UIEB 134

# ---------------------------------------------------------------- stage 3
echo "=== [3/4] ablations: objective, one loss group at a time (§22) ==="
for G in phys cycle color struct spatial jerlov prior; do
  paired "$ABL/no_$G" "$ABL_DATASET" "$ABL_LIMIT" \
         --params "PhysINR-Zero:drop_group=$G"
done

echo "=== [4/4] ablations: runtime, bounds and adaptation length ==="
for S in 1 3; do
  paired "$ABL/stride_$S" "$ABL_DATASET" "$ABL_LIMIT" \
         --params "PhysINR-Zero:loss_stride=$S"
done
for N in 5 20; do
  paired "$ABL/steps_$N" "$ABL_DATASET" "$ABL_LIMIT" \
         --params "PhysINR-Zero:adapt_steps=$N"
done
for T in 2.0 4.0; do
  paired "$ABL/tau_max_$T" "$ABL_DATASET" "$ABL_LIMIT" \
         --params "PhysINR-Zero:tau_max=$T"
done
for D in 0.5 4.0; do
  paired "$ABL/delta_max_$D" "$ABL_DATASET" "$ABL_LIMIT" \
         --params "PhysINR-Zero:delta_max=$D"
done

echo "=== DONE ==="
echo "merge with: python3 experiments/merge_results.py"
