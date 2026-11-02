#!/usr/bin/env bash
# PhysINR-Zero v4 experiments: the main comparison on all five benchmark sets
# and the fifteen ablations, run against the recalibrated level-head
# checkpoint (weights/physinr_zero_v4.pt).
#
# This is the v4 counterpart of run_physinr_zero.sh: same image limits, same
# ablation knobs, same metric families, so the two runs are directly
# comparable and the merge/report pipeline needs no changes.  The ablations
# are test-time overrides on the one checkpoint, so they apply to v4
# unchanged.
#
# Budget (MPS): main comparison ~70 min, ablations ~90 min.  Everything is
# --resume-able, so an interrupt continues rather than restarting.  Never run
# two benchmarks concurrently on this host (see the MPS deadlock notes).
set -u
cd "$(dirname "$0")/.."

export HYDROBENCH_DATA=$PWD/datasets
export HYDROBENCH_MODELS=$PWD/weights
export HYDROBENCH_THIRD_PARTY=$PWD/third_party

PY=python3
OUT=experiments/physinr_results_v4
ABL=experiments/physinr_ablations_v4
METHOD="PhysINR-Zero"
FR="psnr ssim ms_ssim lpips_alex lpips_vgg"
NR="uciqe uiqm uicm uism uiconm ccf"

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

echo "=== v4 main comparison ==="
unpaired "$OUT" UCCS 45
unpaired "$OUT" RUIE 100
paired   "$OUT" EUVP 120
paired   "$OUT" LSUI 120
paired   "$OUT" UIEB 134

echo "=== v4 ablations: objective, one loss group at a time ==="
for G in phys cycle color struct spatial jerlov prior; do
  paired "$ABL/no_$G" "$ABL_DATASET" "$ABL_LIMIT" \
         --params "PhysINR-Zero:drop_group=$G"
done

echo "=== v4 ablations: runtime, bounds and adaptation length ==="
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
echo "merge with: python3 experiments/merge_results.py --v4"
