# Appendix A — Reproducibility and execution

## A.1 Data, environment and seeds

Public source: [GitSkills on Hugging Face](https://huggingface.co/datasets/mvaccargiu/gitskills), fixed revision `ebab17454a7c236f8f26b183567f1a126f42e3f8` (July 2026); the download verifies SHA-256. The environment uses Python 3.12, `pyproject.toml` and `uv.lock`. Seeds: sampling `20261009`; validation 2027, 2028 and 2029; holdout `20261110`; bootstrap 2027. The reference date for age and recency is 2026-07-20. The sample of 200,000 contents is deterministic (`md5(content_hash || ':20261009')`), not a simple random sample. Commands start from the repository root; `make` wraps each step.

## A.2 Data construction and audit

```bash
uv sync
make data      # download, sample, split and count target
make owner     # owner_count, context features, representative audit, EDA
```

Outputs: `data/processed/owner_targets.parquet` (per-content counts on the full population), `context_features.parquet` and, in `results/owner/`, `eda_summary.json`, `eda_target_quantiles.csv`, `eda_owner_concentration.csv`, `audit_by_repository_count.csv`, `audit_by_owner_count.csv`. The EDA manifest records the agreement of the representative rule (97.2%).

## A.3 Main protocol and robustness

```bash
make controls     # control with the original design (Section 4.2)
make primary      # H1–H4 (Section 4.3), cap of 50 per owner
make robustness   # owner thresholds, location classes, alternative target (4.4)
make importance   # permutation importance (4.5)
```

`results/owner/primary/` contains `primary_hypotheses.csv`, the per-fold tables and the exploratory paired tests; `results/owner/robustness/` and `results/owner/models*/` hold each sensitivity, each with a `manifest.json`.

## A.4 Tuning, freezing and holdout

```bash
make final         # nested tuning and freezing (Section 4.6); does not read the holdout
make holdout       # builds the owner-disjoint holdout (Section 3.9)
make holdout-eval  # single evaluation (Section 4.7)
```

The frozen models are in `results/owner/final/final_config_*.json` and were committed before the evaluation (git tag `frozen-before-holdout`). `holdout-eval` writes `results/owner/holdout/evaluation_lock.json` and **refuses** to run again without `--allow-reevaluation`; any re-evaluation must be declared. A reproduction should check `holdout_manifest.json` (zero overlap of owners and contents).

## A.5 Map between tables, figures and artifacts

| Item | Artifact |
|---|---|
| Distribution and concentration (4.1) | `results/owner/eda_target_quantiles.csv`, `eda_owner_concentration.csv` |
| Representative bias (4.2, figure) | `results/owner/audit_by_repository_count.csv`, `eda_owner_and_selection.png` |
| Control with the original design (4.2) | `results/controls/shortcut_summary.csv`, `shortcut_paired.csv` |
| Views and hypotheses (4.3) | `results/owner/primary/{classification,regression}_summary.csv`, `primary_hypotheses.csv` |
| Sensitivity (4.4) | `results/owner/figures/sensitivity_owner_limit.csv`, `results/owner/robustness/*` |
| Importance (4.5) | `results/owner/importance/*_importance.csv` |
| Tuning (4.6) | `results/owner/final/*_nested_summary.csv`, `*_selection_tests.csv` |
| Holdout (4.7) | `results/owner/holdout/holdout_results.json` |

: Origin of the numbers in the report.

## A.6 Checks, deviations and limits

```bash
make test
```

The tests cover the exclusion of contaminated attributes, the path patterns, the paired tests and the Holm correction, the owner bootstrap and the holdout disjointness. Deviations from the protocol: tuning with 2×3 folds and 6 candidates (planned 3×3 and 12) for computational cost; the development machine had 8 GB of memory. Numerical results may vary in the last digits with different library versions. The first version of the study (target `repository_count`, original external test) remains in the repository (`results/count_regression`, `results/discipline_benchmark`) only as history and supports no conclusion of this report. The protocol document `paper/protocol.md` is in Portuguese.
