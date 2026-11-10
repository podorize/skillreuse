# Appendix A — Reproducibility and execution

**Data, environment, seeds.** Public source: [GitSkills on Hugging Face](https://huggingface.co/datasets/mvaccargiu/gitskills), fixed revision `ebab17454a7c236f8f26b183567f1a126f42e3f8` (July 2026; SHA-256 verified). Python 3.12, `pyproject.toml` and `uv.lock`. Seeds: sampling `20261009`; validation 2027–2029; holdout `20261110`; bootstrap 2027. The 200,000-content sample is deterministic (`md5(content_hash || ':20261009')`), not a simple random sample. Code and data tables: [github.com/podorize/skillreuse](https://github.com/podorize/skillreuse).

**Quick reproduction (about 5 minutes, no 6.5 GB download).** `make quick` uses the small derived tables in `data/derived/` (9.5 MB), runs the main protocol with one repetition and re-runs the frozen models on the holdout into a separate directory; the holdout numbers reproduce exactly. A Colab notebook is in `notebooks/reproduce.ipynb`.

**Full pipeline.**

```bash
uv sync
make data owner feature-eda       # download, sample, owner targets, audit, EDA (4.1–4.2)
make controls primary robustness  # original-design control, H1–H4, sensitivities (4.2–4.4)
make importance interpretation    # permutation importance, partial dependence, threshold (4.5, 4.7)
make final holdout                # nested tuning and freezing (4.6); holdout construction (3.9)
make holdout-eval                 # single evaluation (4.7)
```

The frozen models are in `results/owner/final/final_config_*.json`, committed before the evaluation (git tag `frozen-before-holdout`). `holdout-eval` writes `results/owner/holdout/evaluation_lock.json` and refuses to run again without `--allow-reevaluation`; any re-evaluation must be declared. Every table and figure maps to a CSV or JSON under `results/owner/` (and `results/controls/` for the original-design control), each directory with a `manifest.json`; the targets of `make` name the step that produces each one. `make test` runs the 40 tests (excluded attributes, path patterns, paired tests and Holm, owner bootstrap, holdout disjointness, threshold selection).

**Deviations and limits.** Tuning used 2×3 folds and 6 candidates (planned 3×3 and 12) for computational cost, on a machine with 8 GB of memory. Numerical results may vary in the last digits across library versions. The first version of the study (target `repository_count`, original external test) remains in the repository only as history and supports no conclusion of this report. The protocol document `paper/protocol.md` is in Portuguese.
