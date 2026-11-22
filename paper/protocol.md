# Research protocol

---

## 1. Questions and hypotheses

- **RQ1:** *Do the characteristics of a skill's text, observed when it first appears, predict better than chance whether other developers will copy it?*
- **RQ2:** *Which text characteristics carry this signal, and in which direction?*

**Control.** The text could merely reflect where the skill was published: a popular repository attracts copies. We compare a model with attributes of the origin repository **at T0**, including its popularity, and a model with origin repository + text.

| Hypothesis | Criterion (holdout, 95% CI by author bootstrap, Holm across the two) |
|---|---|
| **H1** (RQ1) | AUC of the text model > 0.5 |
| **C1** (control) | AUC(repository + text) − AUC(repository) > 0 |

RQ2 is answered only if H1 is confirmed.

---

## 2. Data

### 2.1 Source, aggregators and population
- GitSkills, revision `ebab17454a7c236f8f26b183567f1a126f42e3f8` (Hugging Face `mvaccargiu/gitskills`), verified by SHA-256. One row per occurrence of a `SKILL.md`; the content hash identifies identical copies.
- **Aggregator:** a repository with **100 or more** distinct contents in GitSkills, such as registries, mirrors and marketplaces that collect skills in bulk. Copies in these repositories do not represent the adoption of the skill by a developer.
- **Population:** contents with at least one occurrence outside aggregators (about 1.16 million).
- **Sample:** the first 150,000 contents of the population in the order of `md5(content_hash || ':20261121')`, with all their occurrences.
- **Snapshot:** S = 09/07/2026, start of the GitSkills crawl.

### 2.2 Collection through the GitHub API
Done once, through the GraphQL API, reading public repositories only, with progress saved in parts so it can be resumed. Network or rate-limit failures are retried up to 3 times, with increasing waits; if they persist, the occurrence receives the status "collection failure". The resulting tables are published with the code.
- **Arrival of each copy** is the later of two dates: (a) the date of the earliest commit, among the 500 most recent commits that touched the file’s path on the default branch, at which the file’s content is exactly the skill’s content hash; and (b) the date the repository was created, which corrects for history inherited from cloned repositories. Arrival is computed for copies in aggregators too, but those are used only in the sensitivity analysis.
- **Status of each occurrence:** dated, repository inaccessible, file missing, hash not found or collection failure.
- **Stars of the origin repository at T0:** count of the stars dated before T0, read in chronological order until passing T0 or reaching 10,000 (recorded as ≥ 10,000; the attribute enters on a log scale).

### 2.3 Definitions
- **T0** = earliest arrival among the dated copies outside aggregators. **Origin** = the copy with that arrival. **Author** = owner of the origin repository.
- **Copied (target)** = there is a dated copy, outside aggregators, by **another owner**, with arrival in [T0, T0 + W].
- **Window W:** fixed in phase 3 by the rule of Section 5, among 30, 60 and 90 days.
- **Eligible** = skills with T0 ≤ S − W, so that all of them have the full observation window.
- **Cap:** at most 50 skills per author, the first ones in hash order, applied before the split.

---

## 3. Attributes

| Group | Use | Columns |
|---|---|---|
| **Text** | Main model (RQ1, RQ2) | 27 attributes of the `SKILL.md`: size, style, Markdown structure, code, links and front matter. Identical in all copies, so they are the text at T0 |
| **Origin repository at T0** | Control only | Stars at T0 (log), age of the repository at T0, language, license, fork, and path of the origin copy (depth, root, template, plugin, vendor, documentation, package, other tools, `.claude`) |
| **Outside the models** | — | Attributes measured at S (current stars and forks, last push, number of skills in the repository); folder and package of the GitSkills representative copy; collection indicators |

---

## 4. Split and validation

- **Split:** by author, over all dated skills after the cap, before any analysis. Within each author-size band (1, 2–3, 4–10, 11–50, more than 50 skills), authors are ordered by `md5(author || ':20261122')`; the first **30%** form the **holdout** and the others the **development set**.
- **Invariants:** no author and no hash on both sides; at most 50 skills per author; 29% to 31% of the authors in the holdout in each band.
- **Validation on the development set:** 3 repetitions × 3 folds of `StratifiedGroupKFold`, grouped by author and stratified by the target (seeds 2027–2029), the same for all algorithms.

---

## 5. Exploratory analysis and preprocessing

- **Development set only:** collection coverage, distribution of T0 and of the time to first copy, types, missing values, outliers, inconsistencies, correlations and relation with the target.
- **Window rule:** among development skills with at least 90 days of observation (T0 ≤ S − 90) and copied within 90 days, W = the smallest value among 30, 60 and 90 that covers at least 80% of the times to first copy. Restricting to skills with 90 days of observation avoids the censoring bias of recent skills.
- **Preprocessing rules:** constant columns removed; skewness > 2 → log in scale-sensitive models; Spearman > 0.95 → one column of the pair removed in those models; impossible values → missing. **Outliers** are kept, because they are real skills (very long texts, a lot of code), and the log reduces their influence in scale-sensitive models. **Discretization** is not used: trees choose their own cut points, and fixed bins would discard information in the other models.
- **Pipelines fitted inside each fold:** median imputation with indicator, or constant category; one-hot with rare categories (< 10) grouped; standardization for logistic regression, k-NN, Naive Bayes and MLP; no resampling; imbalance handled by class weight as a hyperparameter.

---

## 6. Modeling

1. **Spot-checking:** 7 algorithms with default hyperparameters on the text (logistic regression, Naive Bayes, k-NN, decision tree, random forest, HistGradientBoosting, MLP). The 3 with the highest mean AUC go on.
2. **Nested search:** 6 outer folds (2 × 3) and 3 inner folds by author; random search of 10 configurations; default and tuned versions. The highest mean outer AUC wins. If the difference to the second is not significant (Nadeau–Bengio with Holm), the one with the shortest mean training time wins.
3. **Final models** on the whole development set: **text**, **origin repository** and **origin repository + text**, each with its own search.
4. **Simple reference:** the same algorithm with the size attributes only. On the holdout, the AUC difference to the text model is reported with a 95% CI, descriptively (no hypothesis), to show whether the signal goes beyond the length of the text.
5. **Threshold:** maximum balanced accuracy on the out-of-fold predictions of the development set.
6. **Freezing** of models and threshold with hashes, before reading the holdout.

---

## 7. Evaluation

- **Main metric:** ROC AUC. **Secondary:** PR AUC (against the prevalence), Brier score and, at the threshold, precision, recall and balanced accuracy.
- **Development set:** paired comparisons by fold with Nadeau–Bengio and Holm.
- **Holdout:** evaluated **only once**, with a recorded lock; 95% CI by author bootstrap (2,000 resamples).

---

## 8. Sensitivity and robustness

Done after the main evaluation, with the frozen hyperparameters, and reported whatever the result.

| Analysis | Question | How |
|---|---|---|
| **Aggregators** | Does the result depend on the definition of aggregator? | Limit of 500 skills and no exclusion, on the same sample (T0 and target recomputed; text model retrained on the development set and evaluated on the holdout) |
| **Window** | Does the result depend on W? | The other two values of W |
| **Undated copies** | Do inaccessible repositories distort T0 or the target? | Only skills with all copies outside aggregators dated |
| **Temporal split** | Does the model predict future skills? | Cutoff C = 70th percentile of T0 among the eligible development skills; training on development skills with T0 < C; testing on holdout skills with T0 ≥ C |
| **Known variants** | Does performance come from recognizing variants of training skills? | AUC on the holdout only with skills whose front-matter `name` (known at T0) does not appear in the development set. Filters based on the copies are not used, because they depend on the target; the effect of families that circulate together is assessed by the temporal split |

---

## 9. Interpretation (RQ2)

- **Model:** the text model, which uses the winning algorithm and is the best model for the research question. The repository + text model has another purpose (the control) and is not interpreted.
- **Sample:** the whole holdout, or 20,000 skills drawn by author (`md5(author || ':20261123')`) if it is larger.
- **Methods:** permutation importance by family and by attribute, with bootstrap CIs; ALE and SHAP of the 5 main attributes; stability of the family ranking over the 9 folds.
- **Reading:** associations of the model, not causes; only what is stable is interpreted.

---

## 10. Reproducibility

- Python 3.12 with `uv.lock`; fixed seeds; collection tables, derived table and frozen models published; a `Makefile` per phase; tests that check invariants and every number in the paper; a Colab notebook that reproduces the results without the API.

---

## 11. Threats to validity

- **Coverage:** copies deleted before S are not observed, those inaccessible after S are not dated, and GitHub code search only indexes recently active repositories (a small effect in an ecosystem less than a year old); the target is a lower bound and T0 may be late.
- **History:** rewrites (rebase, force-push) are not detected; the arrival rule only corrects histories older than the creation of the repository.
- **Authorship:** the earliest observed copy approximates the author; different accounts of the same person or organization count as different developers; copies made by installers count as adoption.
- **Aggregators:** the limit of 100 skills separates bulk collection from individual adoption only approximately (sensitivity analysis of Section 8).

---

## 12. Phases

| Phase | Content | Output |
|---|---|---|
| 0 | Registration of this protocol | Hash recorded |
| 1 | Aggregators, population, sample and collection (2.1, 2.2) | Collection tables with the status of each occurrence |
| 2 | Authors, split and attributes (2.3, 3, 4) | Invariants approved; manifest |
| 3 | EDA, window and rules (5) | W, eligible skills, target and rules recorded |
| 4 | Spot-checking and nested search (6.1, 6.2) | Winner recorded |
| 5 | Final models, reference and threshold; freezing (6.3–6.6) | Models with hashes |
| 6 | Single evaluation (7) | Results and lock |
| 7 | Sensitivity, robustness and interpretation (8, 9) | Results recorded |
| 8 | Paper, notebook and tests | Numbers checked by the tests |

---

## 13. Deviations

| Date | Phase | Deviation | Reason | Before or after results |
|---|---|---|---|---|
| 17/11/2026 | 1 | **Clarification:** copies whose computed arrival is later than the end of the GitSkills crawl (20/07/2026) receive the status "date after the snapshot" and are not dated (684 of 223,261). A copy observed in the snapshot cannot have arrived after it; the date comes from recreated repositories or rewritten histories. | Data quality; identified in a trial run of the code on partial data | Before the main results |
| 17/11/2026 | 1 | **Execution:** the copies in aggregators, used only in the sensitivity analysis, were collected in batches of 5 with 16 parallel queries, because batches of 40 stalled in repositories with tens of thousands of files. The aggregator sensitivity analysis runs after the other analyses, with the alternative T0 values in a separate file (`data/derived/aggregator_variants.parquet`), without changing the main table. | Execution time | Before the results of that sensitivity analysis |
| 17/11/2026 | 5–6 | **Popularity in the control:** the GitHub API no longer lists stars with their dates (empty query even for popular repositories), so the stars at T0 were all zero and were removed as constant by the rule of Section 5; the original C1 tested only the fixed attributes of the origin repository. The control was redone with **stars and forks of the origin repository at the snapshot** (later than T0, which makes the control more demanding) and only C1 was re-evaluated on the holdout (`results/phase6/control_popularity.json`). The text model, H1 and the original results (`results/phase6/holdout_results.json`) do not change. | Data unavailable; keep the test of the main alternative explanation | After the single evaluation |
| 18/11/2026 | 7 | **Interpretation:** the ALE was computed but not included in the paper; RQ2 is answered with permutation importance and SHAP, which already show the direction and shape of the effects of the main attributes. To keep the repository restricted to what the paper uses, the ALE code and the file `results/phase7/ale_holdout.csv` were removed. | Simplicity of the paper; the ALE repeated what SHAP shows for a linear model | After the interpretation results |
