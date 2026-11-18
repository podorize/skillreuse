# Appendix A — Reproducibility

**Code, data and models.** All code, the tables produced by the collection, the study table, the final models and the result files are available at <https://github.com/podorize/skillreuse>. The data come from a fixed revision of GitSkills (`ebab17454a7c236f8f26b183567f1a126f42e3f8`), downloaded and verified by SHA-256. The environment is Python 3.12 with every dependency pinned in `uv.lock`. All random choices use fixed seeds: `20261121` for the sample, `20261122` for the split, 2027–2029 for the cross-validation folds, 2027 for models, searches and bootstrap, and `20261123` for the interpretation sample.

**Quick reproduction.** The dates of all copies were collected once through the GitHub API and are published, so the study can be reproduced without the API. The Colab notebook `notebooks/reproduce.ipynb` follows this paper section by section: it rebuilds the target from the published tables, re-runs the exploratory rules and the spot-check, applies the saved models to the holdout and checks every number reported here against the result files; automated tests (`make test`) do the same.

**Full pipeline.** From the raw data, the study is reproduced with five commands; `make collect` requires an authenticated GitHub CLI and takes several hours:

```bash
uv sync
make data        # download GitSkills
make collect     # sample and dates (GitHub API)
make study       # split, EDA, selection, models
make holdout     # single holdout evaluation
make robustness  # robustness and interpretation
```

**Collection details.** Of the 223,261 copies outside aggregators, 88.9% were dated; the others were in repositories deleted or made private after the snapshot (9.4%), had lost the file (0.7%), failed after three retries (0.4%), had an arrival after the snapshot (0.3%) or no longer contained the exact content (0.2%). For 11.7% of the dated copies the first commit predated the repository, a sign of inherited history, and the arrival rule moved them to the repository's creation. Of the 84,995 copies in aggregators, collected for the robustness analysis, 86.3% were dated.

**Additional robustness checks.** Two checks complement Section 4.4, both on the holdout with the frozen text model. Restricting the holdout to the 19,862 skills whose copies could all be dated, since an undated copy could hide the true origin or a copy within the window, gives an AUC of 0.626 (95% CI [0.600; 0.649]); these skills are copied less often (5.5%), because a skill with more copies is more likely to have one in a repository that later disappeared. Restricting it to the 15,154 skills (73.7%) whose front-matter name never appears in the development set, to rule out the recognition of variants of known skills, gives 0.622 (95% CI [0.593; 0.651]).

**Safeguards.** The analysis plan was written down before any model was trained (`paper/protocol.md`). The final models were saved with their hashes before the holdout was read, and the holdout evaluation refuses to run a second time unless explicitly forced. Four deviations were recorded in the plan: copies with an arrival after the snapshot are treated as undated; the copies in aggregators were collected later, in smaller batches, because large batches stalled; because the GitHub API no longer lists stargazers with their dates, the popularity of the origin repository at T0 was replaced, after the single evaluation, by its popularity at the snapshot (the original control is kept in `results/phase6/holdout_results.json`); and the planned ALE plots were not reported, since for a linear model they repeat the SHAP dependence plots.
