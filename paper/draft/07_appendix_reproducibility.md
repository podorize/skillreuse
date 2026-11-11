# Appendix A — Reproducibility and execution

**Data, environment, seeds.** Public source: [GitSkills on Hugging Face](https://huggingface.co/datasets/mvaccargiu/gitskills), fixed revision `ebab17454a7c236f8f26b183567f1a126f42e3f8` (July 2026; SHA-256 verified). Python 3.12, `pyproject.toml` and `uv.lock`. Seeds: sampling `20261009`; validation 2027–2029; holdout `20261110`; bootstrap 2027. The 200,000-content sample is deterministic (`md5(content_hash || ':20261009')`). Code and data tables: [github.com/podorize/skillreuse](https://github.com/podorize/skillreuse).

**Quick reproduction (about 5 minutes, no 6.5 GB download).** `make quick` uses the small derived tables in `data/derived/` (9.5 MB), runs the main protocol with one repetition and re-runs the frozen models on the holdout into a separate directory; the holdout numbers reproduce exactly. A Colab notebook is in `notebooks/reproduce.ipynb`.

**Full pipeline.**

```bash
uv sync
make data owner feature-eda
make controls primary robustness
make importance interpretation
make final holdout
make holdout-eval
```

The frozen models are in `results/owner/final/final_config_*.json`, committed before the evaluation (git tag `frozen-before-holdout`). `holdout-eval` writes `results/owner/holdout/evaluation_lock.json` and refuses to run again without `--allow-reevaluation`; any re-evaluation must be declared. Every table and figure maps to a CSV or JSON under `results/owner/` (and `results/controls/`), each directory with a `manifest.json`. `make robustness` also runs the block ablation, and `skillreuse owner-overlap` and `skillreuse fork-audit` reproduce the checks on owner separation (Sections 3.6, 3.9) and forks (Section 5.4). `make test` runs the tests.

**Deviations and limits.** Tuning used 2×3 folds and 6 candidates (planned 3×3 and 12) for computational cost, on a machine with 8 GB of memory. Earlier exploratory results (`results/count_regression`, `results/discipline_benchmark`) remain in the repository only as history. The protocol document `paper/protocol.md` is in Portuguese.
