# 3. Methodology

## 3.1 Research design

The study is empirical, quantitative and predictive, with an interpretive layer; its goal is to measure predictive information, not to estimate causal effects \cite{shmueli2010explain}. The **development** phase (exploratory) includes the dataset audits, the definition of targets and feature views, grouped validation and model selection. The **confirmation** phase evaluates, once, the frozen models on a new holdout, with success criteria fixed beforehand. The full protocol, including the decisions made after seeing results, is in `paper/protocol.md` (in Portuguese).

## 3.2 Data and unit of analysis

The source is the GitSkills corpus \cite{gitskills2027}, revision `ebab17454a7c236f8f26b183567f1a126f42e3f8`: 3,797,117 `SKILL.md` files from 282,200 repositories (July 2026), grouped into 1,877,981 distinct contents. The unit is the content (`content_hash`). All occurrences (repository and path) are kept, but only one **representative** per content receives full text, repository metadata, folder listing and history.

Development uses 200,000 contents chosen by `md5(content_hash || ':20261009')`, split by repository into 160,000 for development and 40,000 for test. We condition on `repository_count >= 2`: 25,918 contents from 7,128 owners in development. In the full population there are 306,088 contents satisfying this condition, of which 230,355 appear under two or more owners. The original test partition was consulted in the first version of the study and does not support confirmatory claims. The reference date for age and recency is 2026-07-20.

## 3.3 Representative rule and audit

The GitSkills documentation indicates that the representative prefers `.claude/skills/`, with a deterministic tie-break \cite{gitskills2027}. We recovered the rule from all raw occurrences: sort by location class (`canonical`, `skills-dir`, `other`) and, on ties, by the smallest pair `(repo_full_name, path)`. It reproduces the representative in 97.2% of the 306,088 contents with at least two repositories, and the class of the representative matches the best available class in 100% of them. The audit compares, by number-of-repositories bin, the fraction of contents whose representative is `canonical` with the fraction of copies that are actually in that class, to establish whether `location_class` and `history_fetched` (history was only collected for standard locations and a sample) can be legitimate predictors.

## 3.4 Targets

| Role | Target | Definition |
|---|---|---|
| Main (classification) | `cross_owner` | `owner_count >= 2` |
| Secondary (regression) | `log1p(owner_count)` | distinct owners across all occurrences |
| Sensitivity | `log1p(repository_count)` | target of the first version |

: Targets. The owner is the prefix of `repo_full_name`.

Counting owners removes self-copying among repositories of the same account. Counts are lower bounds, since GitHub code search indexes only the default branch, files up to 384 KB and recent repositories \cite{gitskills2027}.

## 3.5 Features

| Block | n | Content |
|---|---:|---|
| `skill` | 34 | structure of `SKILL.md` (size, headings, lists, code, links, front matter, scripts, folder) and file name |
| `repo` | 8 | `log1p` of stars and forks, fork flag, age, push recency, active days, language, license |
| `catalog` | 2 | `log1p` of the number of skills in the repository and an indicator of at least 10 |
| `path` | 8 | depth, root, `templates`, `plugins`, `vendor`, `docs`, `packages/src`, other-tool directory |
| *shortcuts* | 4 | `location_class`, `history_fetched`, `commit_count`, `.claude` in the path |

: Feature blocks. The main models use the first four (52 columns, the *clean* view); shortcuts enter only as a finding.

The shortcuts depend on the representative rule and are kept out of the main models. The attributes describe the representative, not all copies; aggregating them over all copies would depend on the same occurrences that define the target and was avoided. Imputation (median with missing indicator), scaling and encoding (one-hot, `min_frequency=10`) are fitted within each fold. The decisions about transformations (`log1p` for linear models only), redundant columns, constant columns and class balance follow from the feature EDA of Section 4.1. We apply no resampling to the 80/20 class imbalance of `cross_owner`: the main metrics (ROC AUC, Brier score, log-loss) are threshold-free, and PR AUC is always read against the prevalence.

## 3.6 Validation and per-owner cap

Validation is grouped by the **owner of the representative repository**: 3 repetitions of 3 folds (`StratifiedGroupKFold`, seeds 2027–2029, strata by `owner_count` bin), with the same partitions for all models. Because a few accounts concentrate the sample, we apply a **cap of 50 contents per owner** (in `content_hash` order), leaving 17,733 contents. The value 50 was chosen *after* observing the sensitivity of the AUC to how concentrated owners are treated (Section 4.4); it is therefore a data-dependent decision. Per fold, we also measure the fraction of validation rows whose front-matter `name` appears in training, and the metrics on the subset without overlap (a joint grouping by owner and `name` forms one component with 64% of the rows and was discarded).

## 3.7 Models, baselines and tuning

The main model is the Random Forest (100 trees for regression, 300 for classification; `max_depth=12`, `min_samples_leaf=3`, `max_features=sqrt`). The linear references are Ridge ($\alpha=10$) and logistic regression ($C=1$); the baselines are the prevalence and the mean and median of `log1p`. Tuning is nested: six families (linear, KNN, tree, Random Forest, Gradient Boosting, HistGradientBoosting) in two variants (default hyperparameters and hyperparameters searched by `RandomizedSearchCV`) on the clean view. The outer validation uses 2 repetitions of 3 owner-grouped folds; the inner search uses 3 owner-grouped folds and 6 candidates. The combination with the best outer mean is frozen (if it is "searched", the search is repeated on the whole development set). The protocol planned 3 repetitions and 12 candidates; we reduced them for computational cost.

## 3.8 Metrics, inference and interpretation

The main metrics are the ROC AUC (classification) and the RMSLE (regression); the secondary ones are PR AUC, Brier score, log-loss, MAE and Spearman. Comparisons are paired by (repetition, fold), with 95% CIs and a t-test with the **Nadeau–Bengio correction** for repeated samples; the paired Wilcoxon test is complementary. Five pre-specified hypotheses form a family corrected by Holm ($\alpha=0.05$); all other comparisons are exploratory.

| ID | Hypothesis | Comparison |
|---|---|---|
| H1 | the clean attributes predict cross-owner reuse | clean RF AUC against 0.5 |
| H2 | the clean attributes reduce the RMSLE of `owner_count` | clean RF against the mean of `log1p` |
| H3a | skill adds to repository and catalog | clean against repository+catalog (AUC) |
| H3b | repository and catalog add to skill | clean against skill (AUC) |
| H4 | the shortcuts add to the clean attributes | clean+shortcuts against clean (RMSLE) |

: Pre-specified hypotheses.

Interpretation uses permutation importance of the clean RF on owner-grouped folds, per column and per block (the columns of a block are permuted together), and out-of-fold partial dependence of the frozen models (trained on each training fold, with the curve computed on the validation rows; mean and standard deviation across folds) for the main attributes. Both are associative; correlated attributes share importance. For the binary task we also report the minority class (`cross_owner = 0`, content that exists under a single owner): its PR AUC against its prevalence and the performance at a decision threshold that maximizes the balanced accuracy on development out-of-fold predictions (3×3 folds). The threshold is fixed on development and then applied, unchanged, to the saved holdout predictions; this is a descriptive reading of the same predictions, not a new evaluation.

## 3.9 New holdout

Because the protocol was fixed after exploratory results, confirmation requires data never consulted. The holdout gathers contents outside the 200,000 sampled, with `repository_count >= 2`, whose owners do not appear among the reused skills of the sample (development and original test), with a cap of 50 per owner: 25,011 contents from 10,484 owners, with no owners or contents in common with the sample and at most 37 contents per owner. Only the prevalence of `cross_owner` (80.8%) was inspected, as an integrity check. The chosen models are trained on the whole development set and evaluated **once**, with CIs by owner bootstrap (2,000 resamples). Success requires the lower AUC bound above 0.5 and the upper bound of the RMSLE difference to the baseline below zero; a lock prevents re-evaluation without explicit declaration.
