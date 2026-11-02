# PhysINR-Zero — HydroBench

**PhysINR-Zero: zero-shot underwater image enhancement with physics-constrained
coordinate fields, evaluated under a leakage-guarded re-benchmarking protocol.**

This repository accompanies the submission to the *IEEE Journal of Oceanic
Engineering*. It contains the complete evaluation code (HydroBench), the
PhysINR-Zero method and its trained checkpoint, the dataset manifests, the
measured results behind every number in the manuscript, and the manuscript
source.

## Authors

| Name | Role | Affiliation | E-mail |
| --- | --- | --- | --- |
| **Ali Hosseini** | First author, corresponding author | Department of Computer Engineering, Shi.C. (Shiraz Branch), Islamic Azad University, Shiraz, Iran | ali.hosseini7148@iau.ac.ir |
| **Saeed Sedighi** | Co-author | Department of Computer Engineering, Shi.C. (Shiraz Branch), Islamic Azad University, Shiraz, Iran | saeed.sedighi@iau.ac.ir |

## Repository layout

```
hydro_bench/            the benchmark framework: runners, metrics, methods,
                        dataset manager, statistics, reporter, tests
  configs/              dataset / method / downstream registries (YAML)
  datasets/manifests/   committed sha256 manifests + splits for every dataset
  methods/classical/    UDCP, Multi-Scale Retinex
  methods/deep/         adapters for Water-Net, FUnIE-GAN, UIEC2-Net,
                        U-Shape Transformer, Five A+, TOPAL, WaterMamba, ...
  methods/deep/physinr_zero/
                        PhysINR-Zero itself: model, physics, losses,
                        meta-training, per-image adaptation, adapter, tests,
                        experiments and run artifacts
results/
  merged_results/       the final per-dataset results.json files behind the
                        paper's tables (all five sets, all eleven methods)
  physinr_results_v4/   the PhysINR-Zero rerun with the display-normalisation
                        stage (PSNR/UCIQE + per-metric tables)
  ablations/            the fifteen-variant UIEB ablation (40 images, per-image)
  figures/              every figure of the manuscript (PDF) + PNG mirrors
  latex/                manuscript_joe.tex/.pdf/.bbl, references.bib,
                        supplementary_joe.tex/.pdf, IEEEtran class/bst
  joe_package/          cover letter, submission checklist + manifest, and
                        PhysINR-Zero_JOE_submission.zip (the upload package)
weights/physinr_zero_v4.pt   the trained checkpoint used for every number in
                        the paper (display_norm=True; 56 KB)
experiments/            the figure/table/statistics build pipeline:
                        merge_results -> make_figures -> make_uieb_tables ->
                        rank_summary -> stats_for_prose -> make_panel_figures,
                        plus build_latex_joe (manuscript) and the audits
paper/02_manuscript/    the Markdown source of the manuscript sections
paper/04_tables/        the generated LaTeX/markdown tables
docs/TECHNICAL_SPEC.md  the full derivation: image-formation model, Jerlov
                        parameterisation, loss groups, hyperparameters
docs/METHODOLOGY_RESOLUTION.md   resolutions of ambiguous specification points
```

## Getting started

```bash
pip install -e ./hydro_bench[torch,lpips,dev]

# 1. sanity: end-to-end on a synthetic dataset, no downloads required
python hydro_bench/run_benchmark.py smoke-test

# 2. point HydroBench at your data (root placeholders resolved from the env)
export HYDROBENCH_DATA=/path/to/datasets
export HYDROBENCH_MODELS=/path/to/weights
export HYDROBENCH_THIRD_PARTY=/path/to/third_party

# 3. verify a dataset against its manifest (checksums + split disjointness)
python hydro_bench/run_benchmark.py verify --dataset UIEB

# 4. run the benchmark
python hydro_bench/run_benchmark.py run --dataset UIEB --split test \
    --methods PhysINR-Zero --metrics psnr ssim uciqe --resume
```

Datasets are **not redistributed** here; each is fetched from its public
source and verified against the committed sha256 manifest before any method
is scored on it (see `hydro_bench/configs/datasets.yaml`).

### PhysINR-Zero standalone

```bash
# smoke: the whole field pipeline end to end, no checkpoint needed
python3 -m hydro_bench.methods.deep.physinr_zero.cli smoke

# enhance images by per-image test-time adaptation
python3 -m hydro_bench.methods.deep.physinr_zero.cli enhance \
    datasets/UIEB/raw_input_imgs/ --output out/ \
    --checkpoint weights/physinr_zero_v4.pt
```

The checkpoint's meta-training pool contains **no underwater photograph** —
the CLI refuses (exit 3) any `--source` that points inside an evaluation
dataset, and the refusal is part of the test suite.

## Reproducing the paper's numbers

```bash
# main results (per-dataset results.json files in results/merged_results/)
python hydro_bench/run_benchmark.py run --dataset UIEB --split test \
    --limit 134 --methods PhysINR-Zero --resume

# ablation: fifteen variants over the first 40 UIEB test images
bash experiments/rerun_ablations_displaynorm.sh

# figures, tables, statistics, then the manuscript itself
python3 experiments/merge_results.py
python3 experiments/make_figures.py
python3 experiments/make_uieb_tables.py
python3 experiments/make_panel_figures.py
python3 experiments/build_latex_joe.py
```

## Tests

```bash
python -m pytest hydro_bench/tests hydro_bench/methods/deep/physinr_zero/tests -q
```

491 tests: metric calibration, manifest tamper detection, split leakage,
runner contracts, deep-adapter contracts (official-weights vs. labelled
fallback), checkpoint/resume semantics, and the zero-shot source guard.

## Headline result

Across 519 images from five benchmark sets against ten baselines, PhysINR-Zero
ranks first of eleven on the no-reference UCIQE measure on four of the five
sets (UIEB 32.186, LSUI 31.618, EUVP 30.988, RUIE 31.803) while trailing the
supervised leaders by 4.5–6.6 dB PSNR on the paired sets — and the paper
quantifies why the two metric families disagree (UICM rewards chroma spread,
the very signal a colour cast carries).

## License

MIT (see the pyproject). Third-party baselines remain under their own
licenses; their code is not redistributed by this repository.
