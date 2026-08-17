# Reproducing the JASA experiments

This directory is the entry point for the numerical results in
`jasa_main20260817.tex` and `jasa_supplement20260817.tex`. Runs are written to
timestamped directories under `reproducibility/runs/`; the CSVs and figures
used by the paper are never overwritten.

## Environment and input check

Use Python 3.8 or newer from the project root:

```bash
python -m pip install -r reproducibility/requirements.txt
python reproducibility/run_all.py --check
```

The check records package versions and verifies the SHA-256 digest of every
curated input. It does not use the network. To additionally parse all eight
manuscript datasets (including the slower CarbonMonitor transformation), run
`python reproducibility/check_reproducibility.py --loaders`.

## Smoke test and full rerun

```bash
python reproducibility/run_all.py --quick
python reproducibility/run_all.py --full
```

Quick mode uses reduced repetitions and updates to test the complete code
path; its numerical output is not intended to match the manuscript. Full mode
uses the defaults reported in the paper and can take many hours. Every run
contains `run_manifest.json`, one log per command, generated CSV files, and
generated figures. Use `--stages` to select any of `synthetic`, `robustness`,
`realdata`, `benchmarks`, and `figures`; use `--dry-run` to print commands.

## Manuscript map

| Result | Generating command or script |
| --- | --- |
| Figure 1 | `code/fig1_ensemble_ode.py` |
| Figure 2(a), null center and universality | `code/exp_4pi_universality.py` |
| Figure 2(b), varying component count and dimension | `code/exp_synthetic_h1_robust.py` |
| Figure 2(c), heteroskedasticity | `code/exp_heterosked_fix_verify.py` |
| Figure 2(d), nonlinear misspecification | `code/exp_misspecification.py` |
| Figure 2 assembly | `code/fig_synthetic_summary.py`, `code/fig_heterosked_misspec.py` |
| Fresh-stream size calculations | `code/exp_type1_rigorous.py` |
| Fresh-stream power calculations | `code/exp_power_H1_rigorous.py` |
| Repeated-data finite-sample audit | `code/exp_finite_sample_reuse.py` |
| Supplementary K=3 and K=4 figures | `code/fig_K34_H0_H1.py` |
| Real-data Stage 1 | `code/exp_realworld_robust.py`, `code/exp_categorical_pooled_remedy.py` |
| Real-data Screen results | `code/exp_realworld_K2_rigorous.py` |
| Xu-style and X-terminal CART discovery benchmarks | `code/exp_sp_benchmark_audit.py` |
| Modified penalized-EM benchmark | `code/exp_kasahara_shimotsu_em_nine.py` |
| Conventional EM--BIC benchmark | `code/exp_conventional_em_detection_nine.py` |
| Nematode application figure | `code/fig_nematode_four_panel_review.py` |

The fixed seeds are defined in the individual scripts and are also written to
each experiment's configuration JSON. Small last-digit differences can arise
from BLAS, NumPy, SciPy, or numba versions; manuscript-scale decisions should
be checked against the precomputed CSVs in `results/`.

## Empirical SP convention

The discovery and verification comparisons record SP when a strict majority
of eligible within-stratum slopes reverse the pooled slope. This majority-sign
rule is motivated by the practical relaxation of the all-strata condition in
Alipourfard, Fennell, and Lerman (2018), *Can You Trust the Trend?*
(https://doi.org/10.1145/3159652.3159684). The Adult Census analysis is not part
of the eight-dataset manuscript comparison.

## Corrected full-design California Housing benchmark

The reproducibility audit found that the historical `--full-x` branch retained
only the first California Housing predictor. The branch is corrected, and two
independent manuscript-scale runs produced byte-identical summary and bootstrap
files with `x_dim=4`, `a_n=8.3`, four starts, and 199 bootstrap samples. The
canonical result is generated with the full benchmark stage:

```bash
python reproducibility/run_all.py --full --stages benchmarks
```
