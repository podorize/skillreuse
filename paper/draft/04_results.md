# 4. Results

## 4.1 The target, the features and owner concentration

In development (25,918 reused contents from 7,128 owners), `owner_count` mirrors `repository_count`: mean 4.50 against 5.08, median 2 in both, P95 15 against 16, maximum 379 against 390, variance-to-mean ratio 23.1 against 22.1. The difference is at the base: **28.0%** of the contents have `owner_count = 1`, so self-copying accounts for 11% of the total mass of `repository_count`. A few owners concentrate the sample: the five largest account for 11.9%, 4.7%, 3.5%, 2.2% and 2.2% of the contents (24.5% together; identifiers omitted), and the largest and the fourth have on average 2.0 repositories but only 1.0 owner per content. In addition, 17.2% of the contents have no front-matter `name`, and 20.1% share it with another content, a sign of variants of the same skill.

**Features.** The clean view has 52 columns (49 numeric, 3 categorical) that expand to 112 after preprocessing. Missing values are rare (13 columns, at most 0.2%) and imputed by the training median with an indicator; one column (`repo_is_fork`) is constant and was kept as harmless. Twenty-five numeric columns have skewness above 2 (for example `sibling_count` 116, `sibling_bytes` 63, reduced to 1.6 and 0.5 by `log1p`); since trees are invariant to monotone transformations, we apply `log1p` only in a sensitivity analysis for linear models (Section 4.4). Eight pairs of columns have |ρ| > 0.95 (for example `body_chars` and `text_char_count`, 0.998); we kept them, since trees tolerate redundancy, and read the importance of redundant columns with caution. No dimensionality reduction was applied. Finally, `cross_owner` is positive for 80.4% of the development contents (80.8% in the holdout), and we do not resample.

## 4.2 Selection bias of the representative

The table compares, on the full population (1,877,981 contents), the fraction of contents whose representative is `canonical` with the fraction of copies that actually are.

| Repositories | contents | representative is `canonical` | copies in `canonical` |
|---|---:|---:|---:|
| 1 | 1,571,893 | 13.3% | 12.6% |
| 2 | 172,624 | 13.8% | 8.5% |
| 3 | 49,535 | 19.5% | 9.1% |
| 4–5 | 36,312 | 24.1% | 10.9% |
| 6–10 | 24,142 | 30.4% | 10.1% |
| 11–50 | 19,589 | 57.7% | 7.7% |
| more than 50 | 3,886 | 91.7% | 5.1% |

: Effect of the representative rule, by number of repositories (full population).

The probability that the representative is `canonical` grows from 14% to 92% with the number of copies, while the actual share of copies in that folder is flat or decreasing (5–13%): a single copy in `.claude/skills/` is enough, by the recovered rule, for the representative to be `canonical`. The representative's location class, and the commit history that exists only for standard locations, **encode the number of copies**. In the first version of the study (target `repository_count`), this inflated the result: with the Random Forest, the repository attributes gave an RMSLE of 0.575, but 0.604 without `history_fetched` and `commit_count` (difference +0.029, 95% CI [0.018; 0.040]), against 0.623 for the mean-of-`log1p` baseline; the location class alone (three categories) gave 0.584.

![Distributions of repositories and owners (left) and effect of the representative rule (right)](../../results/owner/eda_owner_and_selection.png)

## 4.3 Main protocol

With the shortcuts out, 17,733 contents (cap of 50 per owner) and owner-grouped validation, the results are:

| View | RF AUC | logistic AUC | RF RMSLE | Ridge RMSLE |
|---|---:|---:|---:|---:|
| skill | 0.660 ± 0.022 | 0.614 | 0.698 ± 0.007 | 0.713 |
| repository | 0.650 ± 0.019 | 0.655 | 0.707 ± 0.011 | 0.702 |
| repository + catalog | 0.666 ± 0.009 | 0.662 | 0.696 ± 0.010 | 0.699 |
| path + catalog | 0.579 ± 0.019 | 0.576 | 0.718 ± 0.018 | 0.714 |
| **clean (full)** | **0.716 ± 0.024** | 0.680 | **0.679 ± 0.011** | 0.694 |
| clean + shortcuts | 0.717 ± 0.022 | 0.682 | 0.637 ± 0.015 | 0.664 |
| shortcuts only | 0.500 ± 0.015 | 0.510 | 0.688 ± 0.010 | 0.693 |
| *baseline* | 0.500 | 0.500 | 0.720 (mean), 0.780 (median) | |

: Performance by view (mean ± standard deviation over 9 owner-grouped folds).

| | Comparison | Result | 95% CI of the difference | p (Holm) |
|---|---|---|---|---:|
| H1 | clean RF AUC against 0.5 | 0.716 | +0.216 [0.173; 0.259] | < 0.001 |
| H2 | clean RF RMSLE against the mean | 0.679 vs 0.720 | −0.041 [−0.049; −0.033] | < 0.001 |
| H3a | clean against repository+catalog (AUC) | 0.716 vs 0.666 | +0.050 [0.017; 0.083] | 0.008 |
| H3b | clean against skill (AUC) | 0.716 vs 0.660 | +0.056 [0.040; 0.072] | < 0.001 |
| H4 | clean+shortcuts against clean (RMSLE) | 0.637 vs 0.679 | −0.042 [−0.055; −0.029] | < 0.001 |

: Pre-specified hypotheses on development (exploratory until the confirmation of Section 4.7).

All hypotheses are rejected after Holm correction. There is a real but modest signal (AUC 0.72, a 6% RMSLE reduction). Skill and repository contribute **complementarily**: each block alone sits at AUC 0.65–0.67 and the combination at 0.72 (H3a, H3b). The shortcuts add to the regression (H4) but alone do not predict cross-owner reuse (AUC 0.50): their regression gain comes from correlation with the number of copies, not from cross-owner reuse.

## 4.4 Robustness

| Treatment of concentrated owners | contents | clean RF AUC | RMSLE gain (clean) |
|---|---:|---:|---:|
| no limit | 25,918 | 0.747 ± 0.140 | 0.039 |
| exclude owners with more than 1000 | 21,615 | 0.615 ± 0.036 | 0.036 |
| exclude owners with more than 500 | 19,567 | 0.708 ± 0.037 | 0.030 |
| exclude owners with more than 200 | 18,443 | 0.732 ± 0.035 | 0.037 |
| **50 per owner** | 17,733 | 0.716 ± 0.024 | 0.041 |

: Sensitivity to the treatment of concentrated owners (RF; gain relative to the mean-of-`log1p` baseline).

The RMSLE gain is stable (0.03–0.04), but the AUC oscillates from 0.615 to 0.747: a few owners with 500 to 1000 contents lower it by 0.1, and with no limit one fold concentrates one owner and the standard deviation reaches 0.14. The cap of 50 stabilizes the estimates but is a data-dependent choice.

Other checks. (i) Fixing the location class and training within each one (owners with more than 500 contents excluded), the clean RF still beats the baseline: `canonical` (5,439 contents) RMSLE −0.141, AUC 0.714; `skills-dir` (10,721) −0.013, 0.672; `other` (3,810) −0.011, 0.828. (ii) With `repository_count` as target, the gain is the same (−0.031 against −0.030). (iii) Only 18.8% of validation rows have a `name` seen in training, and the gain persists without them (RMSLE −0.039 against −0.041; AUC 0.726 against 0.716). (iv) Repository-grouped validation gave no larger estimates (RMSLE 0.657 against 0.655; AUC 0.735 against 0.747). (v) Applying `log1p` to the 25 skewed columns improves the linear models only slightly (logistic AUC 0.680 → 0.691; Ridge RMSLE 0.694 → 0.690) and leaves the forests unchanged; the forest's advantage over the linear model remains indistinguishable from zero (AUC +0.023, 95% CI [−0.012; 0.058]).

## 4.5 Interpretation

For cross-owner reuse, the repository block weighs more (AUC drop of 0.104 when permuted) than skill (0.084), catalog (0.016) and path (≈ 0); among columns, `log_stars` dominates (0.056), followed by `log_forks` (0.021) and `push_recency_days` (0.016). In regression, repository (0.019) and skill (0.018) weigh similarly, and the largest column is the license (0.013), followed by catalog size (0.011) and path depth (0.005). Uncertainties are large (6 folds), and correlated attributes share importance.

![Out-of-fold partial dependence of the frozen models: classification RF (top) and regression HistGB (bottom); bands are ±1 standard deviation across folds, dashed lines the mean observed prediction](../../results/owner/figures/pdp.png)

The partial dependence shows the **direction**. The probability of cross-owner reuse rises with stars (0.78 with none to 0.85 at about 540, mostly between 1 and 12 stars) and forks (0.80 to 0.83), and also with the **time since the last push**: repositories pushed more than about 100 days ago have higher probability (0.79 at 4 days, 0.85 at 176). In regression, the expected number of owners rises with catalog size (clearly only above roughly 60 skills) and with path depth (a jump from 2 to 3). The unexpected result is the license: what matters is its **absence**. Repositories without a license (52% of the development contents, median of zero stars) host skills with 6.6 owners on average, against 3.7 for MIT, and the model predicts 1.48 against 1.29–1.30 for `log1p(owner_count)` for any license; they also have more `canonical` skills (34% against 24%). We do not know the mechanism: it may reflect repositories created automatically or by installers, and the data do not separate this from an effect of the missing license itself.

## 4.6 Model selection and tuning

| Family | AUC default | AUC searched | RMSLE default | RMSLE searched |
|---|---:|---:|---:|---:|
| linear | 0.678 | 0.689 | 0.692 | 0.690 |
| KNN | 0.667 | 0.690 | 0.703 | 0.686 |
| tree | 0.614 | 0.634 | 0.807 | 0.699 |
| Random Forest | **0.716** | 0.701 | 0.679 | 0.675 |
| Gradient Boosting | 0.691 | 0.704 | 0.690 | 0.671 |
| HistGradientBoosting | 0.693 | 0.682 | **0.670** | 0.688 |
| *baseline* | 0.500 | | 0.720 | |

: Nested validation (mean of 6 outer owner-grouped folds; 6 candidates per search).

By the pre-specified rule we froze the default RF (classification) and the default HistGradientBoosting (regression). **The families tie:** the RF against the runner-up (searched Gradient Boosting) differs by 0.012 AUC (95% CI [−0.003; 0.026]), and in regression the HistGB differs from the searched Gradient Boosting by −0.001 [−0.027; 0.024]. Only the tree (and, with less evidence, the default KNN) falls behind, and in classification even these differences do not survive Holm correction (p between 0.06 and 0.13). The search did not beat the default configuration (RF: 0.716 against 0.701).

## 4.7 Confirmation on the holdout

The frozen models were evaluated once on 25,011 contents from 10,484 owners disjoint from development.

| Task | Model | Result | Baseline | 95% CI (owner bootstrap) |
|---|---|---:|---:|---|
| `cross_owner` | Random Forest | AUC **0.731**; PR AUC 0.913; Brier 0.140; log-loss 0.444 | 0.5; 0.808; 0.155; 0.490 | AUC [0.716; 0.745]; PR AUC [0.905; 0.921] |
| `log1p(owner_count)` | HistGradientBoosting | RMSLE **0.625**; Spearman 0.330 | 0.664 (mean of `log1p`) | difference −0.038 [−0.044; −0.032] |

: Single evaluation on the holdout.

![ROC curve, calibration and precision–recall curve of the single-owner (minority) class for the RF on the holdout](../../results/owner/figures/holdout_roc_calibration.png)

Both success criteria were met, and the holdout AUC (0.731) is consistent with the nested estimate on development (0.716). The holdout has a different composition (36% `canonical` skills against 21% in development, 9% `other` against 21%), since concentrated owners are left out by construction; H3 and H4 were not re-evaluated and remain exploratory.

**Minority class and threshold.** The PR AUC above refers to the majority class (80.8% prevalence), so it says little. For the minority class (content under a single owner, 19.2%), the PR AUC is 0.390 (95% CI [0.362; 0.418]), about twice its prevalence. The threshold that maximizes the balanced accuracy on development out-of-fold predictions is 0.80 (the default 0.5 gives only 0.506 there). Applied unchanged to the holdout, it gives a balanced accuracy of 0.659 (95% CI [0.647; 0.670]), against 0.663 on development, with a recall of 0.79 and a precision of 0.28 for the single-owner class (3,816 of 4,809 true single-owner contents recovered, but 9,607 cross-owner contents flagged as single-owner). The model therefore **orders** contents usefully, but a threshold produces many false alarms, and it should not be used for automated decisions.
