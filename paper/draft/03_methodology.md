# 3. Methodology

## 3.1 Research design

The study is empirical, quantitative and predictive, with an interpretive layer; it measures predictive information and does not estimate causal effects \cite{shmueli2010explain}. The **development** phase (exploratory) comprises the dataset audits, the definition of targets and feature views, grouped validation and model selection. The **confirmation** phase evaluates, once, the frozen models on a new holdout, with success criteria fixed beforehand. The full protocol, including decisions made after seeing results, is in `paper/protocol.md` (Portuguese).

## 3.2 Data and unit of analysis

The source is GitSkills \cite{gitskills2027}, revision `ebab17454a7c236f8f26b183567f1a126f42e3f8`: 3,797,117 `SKILL.md` files from 282,200 repositories (July 2026), grouped into 1,877,981 distinct contents (`content_hash`). All occurrences are kept, but only one **representative** per content receives full text, repository metadata, folder listing and history. Development uses 200,000 contents chosen by `md5(content_hash || ':20261009')`, split by repository into 160,000 for development and 40,000 for test; we condition on `repository_count >= 2`, which leaves 25,918 contents from 7,128 owners in development. In the full population, 306,088 contents satisfy this condition, 230,355 of them under two or more owners. The original test partition was consulted in the first version of the study and supports no confirmatory claim. The reference date for age and recency is 2026-07-20.

## 3.3 Representative rule and audit

GitSkills documents that the representative prefers `.claude/skills/`, with a deterministic tie-break \cite{gitskills2027}. We recovered the rule from all raw occurrences: sort by location class (`canonical`, `skills-dir`, `other`) and, on ties, by the smallest `(repo_full_name, path)`. It reproduces the representative in 97.2% of the 306,088 contents with at least two repositories, and the representative's class equals the best available class in 100% of them. The audit compares, by number of repositories, the fraction of contents whose representative is `canonical` with the fraction of copies that actually are, to decide whether `location_class` and `history_fetched` (history was collected only for standard locations and a sample) can be legitimate predictors.

## 3.4 Targets

The main target is `cross_owner = owner_count >= 2`, where `owner_count` is the number of distinct owners (prefix of `repo_full_name`) across all occurrences; the secondary target is `log1p(owner_count)`, and `log1p(repository_count)` is used for sensitivity. Counting owners removes self-copying among repositories of one account. Counts are lower bounds, since GitHub code search indexes only the default branch, files up to 384 KB and recent repositories \cite{gitskills2027}.

## 3.5 Features

| Block | n | Content |
|---|---:|---|
| `skill` | 34 | structure of `SKILL.md` (size, headings, lists, code, links, front matter, scripts, folder) and file name |
| `repo` | 8 | `log1p` of stars and forks, fork flag, age, days since last push, active days, language, license |
| `catalog` | 2 | `log1p` of the number of skills in the repository; indicator of at least 10 |
| `path` | 8 | depth, root, `templates`, `plugins`, `vendor`, `docs`, `packages/src`, other-tool directory |
| *shortcuts* | 4 | `location_class`, `history_fetched`, `commit_count`, `.claude` in the path |

: Feature blocks. The main models use the first four (52 columns, the *clean* view); shortcuts enter only as a finding.

The shortcuts depend on the representative rule and are kept out of the main models. The attributes describe the representative, not all copies; aggregating over all copies would use the same occurrences that define the target and was avoided. Imputation (median with missing indicator), scaling and one-hot encoding (`min_frequency=10`) are fitted within each fold. Decisions on transformations (`log1p` for linear models only), redundant and constant columns and class balance follow from the feature EDA (Section 4.1). We do not resample the 80/20 imbalance of `cross_owner`: the main metrics are threshold-free, and PR AUC is read against prevalence.

## 3.6 Validation and per-owner cap

Validation is grouped by the **owner of the representative repository**: 3 repetitions of 3 folds (`StratifiedGroupKFold`, seeds 2027–2029, strata by `owner_count` bin), with the same partitions for all models. Because a few accounts concentrate the sample, we cap the contribution at **50 contents per owner** (in `content_hash` order), leaving 17,733 contents. The value 50 was chosen *after* observing the AUC's sensitivity to how concentrated owners are treated (Section 4.4); it is a data-dependent decision. Per fold we also measure the share of validation rows whose front-matter `name` appears in training and the metrics on the subset without overlap (a joint owner–`name` grouping forms a component with 64% of the rows and was discarded).

## 3.7 Models, baselines and tuning

The main model is the Random Forest (100 trees for regression, 300 for classification; `max_depth=12`, `min_samples_leaf=3`, `max_features=sqrt`), with Ridge ($\alpha=10$) and logistic regression ($C=1$) as linear references and the prevalence and the mean and median of `log1p` as baselines. Tuning is nested: six families (linear, KNN, tree, Random Forest, Gradient Boosting, HistGradientBoosting) in two variants (default and `RandomizedSearchCV`) on the clean view, with an outer validation of 2 repetitions of 3 owner-grouped folds and an inner search of 3 owner-grouped folds and 6 candidates (planned: 3 repetitions and 12 candidates; reduced for computational cost). The combination with the best outer mean is frozen.

## 3.8 Metrics, inference and interpretation

The main metrics are ROC AUC (classification) and RMSLE (regression); the secondary ones are PR AUC, Brier score, log-loss, MAE and Spearman. Comparisons are paired by (repetition, fold), with 95% CIs and a t-test with the **Nadeau–Bengio correction** for repeated samples. Five pre-specified hypotheses form a family corrected by Holm ($\alpha=0.05$); all other comparisons are exploratory: **H1** the clean attributes predict cross-owner reuse (AUC against 0.5); **H2** they reduce the RMSLE of `owner_count` (clean RF against the mean of `log1p`); **H3a/H3b** skill adds to repository+catalog and vice versa (AUC); **H4** the shortcuts add to the clean attributes (RMSLE).

Interpretation uses permutation importance of the clean RF on owner-grouped folds (per column and per block) and out-of-fold partial dependence of the frozen models (trained on each training fold, curves computed on the validation rows). Both are associative. For the binary task we also report the minority class (`cross_owner = 0`) and the performance at a threshold that maximizes the balanced accuracy on development out-of-fold predictions; the threshold is then applied, unchanged, to the saved holdout predictions, as a descriptive reading and not a new evaluation.

## 3.9 New holdout

The holdout gathers contents outside the 200,000 sampled, with `repository_count >= 2`, whose owners do not appear among the reused skills of the sample (development and original test), with the same cap of 50: 25,011 contents from 10,484 owners, sharing no owners or contents with the sample (at most 37 contents per owner). Only the prevalence of `cross_owner` (80.8%) was inspected, as an integrity check. The chosen models are trained on the whole development set and evaluated **once**, with CIs by owner bootstrap (2,000 resamples). Success requires the lower AUC bound above 0.5 and the upper bound of the RMSLE difference to the baseline below zero; a lock prevents re-evaluation without explicit declaration.
