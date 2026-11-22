# 3. Materials and methods

This section follows the order of the work: we first built the data (3.1–3.4), then explored and prepared it (3.5), developed the models (3.6), evaluated them (3.7) and interpreted the final one (3.8). The whole analysis plan was written down before any model was trained.

## 3.1 Data

We use the GitSkills dataset \cite{gitskills2027}, published on Hugging Face, at a fixed revision to ensure reproducibility across all analyses.
The dataset contains 3,797,117 `SKILL.md` files from 282,200 public GitHub repositories. Identical files are grouped by content hash, leaving 1,877,981 distinct contents.
In this study, each distinct content is defined as a skill, and its occurrences across repositories are considered separate copies.

Some repositories collect skills in bulk rather than hosting them as a result of individual developer decisions. These include registries, mirrors, and marketplaces, with the largest containing 89,908 distinct skills.
We classify repositories containing at least 100 distinct skills as aggregators (2,867 repositories).
Since copies in aggregators reflect bulk collection rather than individual adoption, these repositories are excluded from the main analysis.
This leaves 1,163,728 skills with at least one occurrence in a non-aggregator repository.

Determining when each skill was copied requires querying its file history through the GitHub API, which imposes hourly rate limits (Section 3.2).
The 1,163,728 skills have a total of 2,395,296 copies.
Dating all copies would require more than two days of continuous API queries, whereas dating the copies in our random sample took approximately seven hours. 
We therefore selected a random sample of 150,000 skills, comprising 308,256 copies: 223,261 in non-aggregator repositories and 84,995 in aggregators.

## 3.2 Measuring whether a skill was copied

GitSkills shows *where* each skill was in July 2026, but not *when* it got there. This is a problem for our question in two ways. First, without dates we cannot tell the original from its copies: if a skill is in three repositories, we do not know which one published it and which ones copied it. Second, skills of different ages cannot be compared: a skill published a year before the snapshot has had far more time to be copied than one published the week before, so a count of copies in a snapshot mixes the appeal of a skill with its age.

We therefore dated the 223,261 copies of the sampled skills that are outside aggregators (the 84,995 copies in aggregators were dated only for a robustness analysis, Section 3.7). For each copy, we asked the GitHub API for the history of the file and recorded the first commit in which the file had exactly the skill's text. We call this date the **arrival** of the copy. When a repository is created by cloning another one, it inherits the old history and its first commit is older than the repository itself; in that case the arrival is the date the repository was created. Copies that could not be dated, mostly because their repository was deleted after July 2026, were ignored.

With these dates we can follow each skill from the moment it first appeared. The earliest arrival is the skill's first appearance, or **T0**; the copy that arrived first is the **origin**; and the owner of the origin repository is the skill's **author**. A skill is **copied** — the target of our models, and the form of replication we study — if another developer's copy, outside aggregators, arrived within **W days after T0**. Copies by the author's own account do not count.

*Example.* A skill first appears in Ana's repository on 2 March 2026, so T0 is 2 March and Ana is the author. Bruno copies it on 12 March. Ana also copies it into a second project of her own, and a mirror collects it in April. With W = 60 days, the skill counts as copied, because of Bruno's copy; Ana's second copy and the mirror do not count.

The window W makes the comparison fair: every skill gets the same amount of time to be copied, whatever its age. W was chosen during the exploratory analysis by a rule written in advance: the shortest of 30, 60 or 90 days that captures at least 80% of first copies, measured only on skills that had been visible for at least 90 days. A skill can only be used if its whole window ends before the snapshot of 9 July 2026; we call these skills **eligible**.

## 3.3 Attributes

From the content of a skill’s Markdown file, we extracted the 27 attributes, listed in Table 1: measures of size, writing style, structure, code, links and front matter (the header with the skill's name and description).
Because all copies of a skill share identical text, these attributes represent the information available to other developers when deciding whether to copy it and do not incorporate future information.
GitSkills also describes the folder and companion files of one representative copy of each skill, but that copy is chosen by a rule that depends on how many copies exist, so its attributes would reveal the answer; we do not use them.

| Group | Columns | Description |
|----------------------|-----------:|-------------------------------------------------------------------|
| Size | 7 | Characters, lines, non-empty lines, words, unique words, sentences, body length |
| Text style | 3 | Lexical diversity (unique words over all words), average word length, average words per sentence |
| Markdown structure | 7 | Headings and their depth, bulleted and numbered items, table rows, quotes, checkboxes |
| Code | 3 | Code blocks, inline code, shell prompts |
| Links | 2 | Links and URLs |
| Front matter | 5 | Whether it is valid; length and absence of the name and of the description |
| Origin repository (control only) | 14 | Age of the repository at T0, language, license, fork flag, folder of the file (depth, root, template, plugin, vendor, documentation, package, other-tool and `.claude` folders), stars and forks |

: Overview of the extracted attributes. The 27 text attributes constitute the main feature set, while origin-repository attributes are used only as controls

A natural objection is that skills from popular repositories may be copied simply because of where they are. To test it, a second group of attributes describes the **origin repository** (Table 1, last row). We planned to measure its popularity by the stars it had at T0, but the GitHub API no longer reports when each star was given. We used instead the stars and forks of the repository in July 2026. This popularity was measured after T0 and may even be partly caused by the copies, so it gives the repository an advantage: if the text still adds information in this comparison, the conclusion only becomes stronger.

## 3.4 Training and test data

The models must work for developers they have never seen. Skills by the same author tend to look alike, so if one author's skills appeared in both training and test data, a model could simply recognize the author. We therefore split the data **by author**, before any analysis. Each author contributes at most 50 skills, so that a few prolific accounts cannot dominate. A random 30% of the authors form the **holdout**, the test set used only once at the end; the other 70% form the **development set**, used for everything else. The draw is made separately for authors with 1, 2–3, 4–10 and 11–50 skills, so both sets have the same mix of small and large accounts.

## 3.5 Exploratory analysis and preprocessing

The exploratory analysis used only the development set, so that nothing from the test set could influence our choices. We studied the target (how many skills are copied, and how fast) and each attribute: its type and distribution, missing values, outliers, inconsistencies, its correlation with the other attributes, and how the share of copied skills changes across its quartiles (four groups of equal size). Each quartile holds about 12,000 skills, about 930 of them copied, so the shares are stable.

Four rules, written in advance, turned these findings into preprocessing steps (Table 2). Constant columns carry no information and are removed. Counts that are very skewed — most skills have no links or code, while a few have thousands — would let a handful of extreme skills dominate linear and distance-based models, so columns with skewness above 2 are log-transformed for those models; the extreme values are kept, because they are real skills, not errors. Near-duplicate columns (Spearman correlation above 0.95) make the coefficients of linear models unstable, so those models keep only one column of each pair. Impossible values are treated as missing. We did not resample the data to correct the imbalance between copied and other skills: class weights do the same job inside the learning algorithm and were tuned as a hyperparameter. All steps are fitted inside each training fold, in scikit-learn pipelines \cite{pedregosa2011scikit}, so that no information from the validation data leaks into training.

| Step | Decision (and the finding that motivated it) |
|----------------------|--------------------------------------------------------------------|
| Missing values | Median imputation with indicator (numeric), constant category (categorical); only 0.02% missing |
| Categorical encoding | One-hot, categories with fewer than ten occurrences grouped; long tails of languages and licenses |
| Normalization | Standardization for scale-sensitive models, none for trees; very different scales |
| Outliers | Kept, with log transform for scale-sensitive models; 22 columns with skewness above 2 (Figure 2a) |
| Discretization | Not used; trees choose their own cut points, and bins would discard information |
| Class imbalance | No resampling, class weight tuned, threshold-free main metric; 7.7% of skills copied |
| Dimensionality | One column of each near-duplicate pair removed for scale-sensitive models; eight pairs above 0.95 (Figure 2b) |

: Preprocessing decisions and the findings that motivated them.

## 3.6 Model development

**Validation.** Every algorithm was trained and validated on exactly the same partitions of the development set: 3-fold cross-validation, repeated three times with different seeds. The folds are grouped by author, for the same reason as the split, and stratified so that each fold has about the same share of copied skills. Each configuration therefore gets nine estimates, which averages out the luck of a single partition.

**Spot-check.** We first compared seven algorithms with their default settings, chosen to represent different ways of learning: a linear model (logistic regression), a probabilistic model that treats attributes as independent (Gaussian Naive Bayes), a model based on similar examples (k-nearest neighbors), a single tree of rules (decision tree), two ensembles of trees — one that averages independent trees (random forest) and one that adds trees that correct each other (HistGradientBoosting) — and a neural network (multilayer perceptron). This quick comparison shows which kinds of models suit the problem before any effort goes into tuning.

**Tuning and selection.** The three best algorithms were then tuned with nested cross-validation. In each of six outer folds, a random search tried ten hyperparameter configurations (Table 3) on three inner folds and kept the best one; the outer fold then measured how well it performed on data the search had not seen. Each algorithm was also evaluated with its default settings. The configuration with the highest average AUC wins. If it is not significantly better than the others, the fastest of the tied configurations wins, so that we only accept extra complexity when it pays off.

| Algorithm | Hyperparameters searched (10 random configurations) |
|------------------------------|----------------------------------------------------------|
| HistGradientBoosting | learning rate 0.02–0.2 (log); 100–500 iterations; 8–64 leaves; 5–100 samples per leaf; L2 penalty 0.001–10 (log); class weight none or balanced |
| Logistic regression | inverse regularization C 0.001–100 (log); class weight none or balanced |
| Multilayer perceptron | hidden layers (64), (128) or (64, 32); L2 penalty $10^{-5}$–$10^{-2}$ (log); initial learning rate $10^{-4}$–$10^{-2}$ (log); early stopping |

: Search spaces of the three algorithms selected by the spot-check.

**Final models.** The selected configuration was trained on the whole development set with the text attributes. This **text model** answers RQ1 and is the one we interpret for RQ2. For comparison, the same configuration was trained with three other sets of attributes: **size only**, to see whether the text says more than its length; the **origin repository**; and the **origin repository plus the text**. We also fixed a decision threshold, needed only for metrics such as precision and recall, as the one that maximizes balanced accuracy on the development set. All models and the threshold were saved, with a fingerprint of each file, before the test set was opened.

## 3.7 Evaluation

**Metrics.** Our main metric is the area under the ROC curve (AUC). It is the probability that the model gives a higher score to a randomly chosen copied skill than to a randomly chosen skill that was not copied. It matches our question — whether the text *ranks* skills by their chances of being copied — and it does not depend on a threshold or on how rare copied skills are. Because copied skills are rare, we also report the area under the precision–recall curve (PR AUC), which must be compared with the share of copied skills (what a random model would get); the Brier score, which measures whether predicted probabilities are realistic; and, at the chosen threshold, precision, recall, balanced accuracy and the confusion matrix.

**Statistics.** On the development set, two models are compared fold by fold. Because the training sets of the folds overlap, the nine results are not independent and an ordinary paired test would be too optimistic; the Nadeau–Bengio correction accounts for this overlap \cite{nadeau2003inference}, and Holm's method adjusts for several comparisons made at once \cite{holm1979simple}. The test set was evaluated **once**. Its 95% confidence intervals come from 2,000 bootstrap samples drawn by author, since skills of the same author are not independent. RQ1 is answered positively if the interval of the text model's AUC lies above 0.5, the value of a random guess. The control is passed if the interval of the gain from adding the text to the origin repository lies above zero.

**Robustness.** Three further analyses, also planned in advance, check whether the result depends on our own choices: a different window (30 or 90 days), a different definition of aggregator (500 skills instead of 100, or no exclusion at all, which required dating the 84,995 copies in aggregators too), and a **split in time**, in which the model learns from skills that appeared earlier and is tested on skills that appeared later. The last one is the closest test of predicting the future.

## 3.8 Interpretation

To answer RQ2 we interpret one model, the text model, on 20,000 skills of the test set. We use two complementary methods. **Permutation importance** shuffles the values of an attribute and measures how much the AUC drops: the larger the drop, the more the model relies on that attribute \cite{breiman2001statistical,molnar2020interpretable}. Because many attributes are correlated, we also shuffle whole families of Table 1 at once and read importance mainly at that level. **SHAP values** split each prediction into the contribution of each attribute \cite{lundberg2017unified}: they show in which direction an attribute pushes the prediction and how that effect changes with the attribute's value. For a logistic regression, the SHAP value of an attribute is simply its coefficient times how far the skill is from the average on that attribute, so SHAP turns the coefficients into effects on individual skills. Finally, we repeat the permutation on the nine development folds to check that the ranking is stable.
