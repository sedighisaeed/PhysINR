# PhysINR-Zero migration — 2026-10-01

PhysINR-Zero moved from a repository-root package into the benchmark's method
tree so it lives alongside the other baseline methods. Nothing was deleted;
every old entry point keeps working.

## What moved where

| Old path | New path |
|---|---|
| `physinr_zero/{model,physics,config,losses,synthesize,meta_train,adapt,cli,utils}.py` | `hydro_bench/methods/deep/physinr_zero/` |
| `physinr_zero/tests/` | `hydro_bench/methods/deep/physinr_zero/tests/` |
| `physinr_zero/__init__.py` (package doc) | merged into `hydro_bench/methods/deep/physinr_zero/__init__.py` |
| `experiments/{v2_eval,v3_eval,v4_revision_eval,type_head_eval,type_head_identifiability,gains_source_ab,module_ablation,probe_tau_meta_init,make_revision_grid,make_cast_refactor_grid}.py` | `hydro_bench/methods/deep/physinr_zero/experiments/` |
| `experiments/{run_physinr_zero.sh,run_physinr_zero_v4.sh,run_revision_programme.sh}` | `hydro_bench/methods/deep/physinr_zero/experiments/` |
| `logs_v3/`, `logs_v4/`, `logs_v3_meta_train.txt`, `logs_v4_meta_train.txt`, `hydro_bench_results_v4/` | `hydro_bench/methods/deep/physinr_zero/artifacts/` |

## What deliberately did not move

- `weights/physinr_zero_*.pt` — every method's checkpoints live in the shared
  `weights/` directory (`methods.yaml` resolves checkpoints from there).
- `hydro_bench/methods/deep/physinr_zero/adapter.py` — the benchmark adapter
  was already in the method tree; the implementation joined it there.
- `experiments/{merge_results,make_figures,make_uieb_tables,build_latex,
  check_manuscript_numbers}.py` — shared benchmark/paper pipeline used by all
  methods.
- `paper/` — the submission package describes the whole benchmark; its
  provenance records also preserve the old paths as history.
- `runs/`, `hydro_bench_results*/` (other methods' results) — untouched.

## Compatibility

- `physinr_zero/__init__.py` remains at the repository root as a shim that
  aliases the old module names to the new home. It exists because the
  published checkpoints (`weights/physinr_zero_*.pt`) were saved with
  `torch.load(..., weights_only=False)` and pickle classes under the old
  `physinr_zero.model` paths.
- New code should import `hydro_bench.methods.deep.physinr_zero`.
- `methods.yaml` was already pointing at the adapter's permanent location and
  needed no change.
- `REPO`/`sys.path` anchors inside the moved scripts were re-anchored for the
  new depth (`parents[4]`/`parents[5]`); the shell scripts are still run from
  the repository root.

## Verification after the move

- 219 tests pass (2 skipped, as before the move):
  `python3 -m pytest hydro_bench/methods/deep/physinr_zero/tests/
  hydro_bench/tests/test_physinr_zero_adapter.py`
- `weights/physinr_zero_v4.pt` loads (`torch.load`, CPU) through the shim.
- `hydro_bench/configs/methods.yaml` adapter import unchanged.
