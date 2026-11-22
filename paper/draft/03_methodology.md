# 3. Materials and methods

This section follows the study workflow: we first built the dataset (3.1–3.4), then explored and prepared it (3.5), developed the models (3.6), evaluated their performance (3.7), and interpreted the final model (3.8). The analysis plan was established before model training.

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

From each skill's Markdown file, we extracted 27 attributes describing its size, writing style, structure, code, links, and front matter (the header containing the skill's name and description):

- Size (7): character count, line count, non-empty line count, word count, unique word count, sentence count, and body length.

- Text style (3): lexical diversity (unique words over all words), average word length, and average words per sentence.

- Markdown structure (7): headings and their depth, bulleted and numbered items, table rows, quotes, and checkboxes.

- Code (3): code blocks, inline code, and shell prompts.

- Links (2): links and URLs.

- Front matter (5): validity, length, and absence of the name and of the description.

Because all copies of a skill share identical text, these attributes represent the information available to other developers when deciding whether to copy it, without incorporating future information.
GitSkills also describes the folder and companion files of one representative copy of each skill, but that copy is chosen by a rule that depends on how many copies exist, so its attributes would reveal the answer; we do not use them.

We also extracted repository-level attributes as controls to account for the possibility that copying is driven by repository visibility rather than skill content.
These 14 attributes cover repository age at T0, language, license, fork flag, folder of the file (depth, root, template, plugin, vendor, documentation, package, other-tool and `.claude` folders), stars and forks.
We initially intended to measure popularity at T0, but the GitHub API no longer provides the dates on which stars were given.
We therefore used star and fork counts from July 2026. Although measured after T0, these counts may partly reflect copying activity, making this a conservative comparison: if skill-text attributes still provide additional predictive information, this supports their contribution beyond repository characteristics.

## 3.4 Training and test data

The models must work for developers they have never seen. Skills by the same author tend to look alike, so if one author's skills appeared in both training and test data, a model could simply recognize the author. We therefore split the data **by author**, before any analysis. Each author contributes at most 50 skills, so that a few prolific accounts cannot dominate. A random 30% of the authors form the **holdout**, the test set used only once at the end; the other 70% form the **development set**, used for everything else. The draw is made separately for authors with 1, 2–3, 4–10 and 11–50 skills, so both sets have the same mix of small and large accounts.

## 3.5 Exploratory analysis and preprocessing

The exploratory analysis used only the development set, so that nothing from the test set could influence our choices. We studied the target (how many skills are copied, and how fast) and each attribute: its type and distribution, missing values, outliers, inconsistencies, its correlation with the other attributes, and how the share of copied skills changes across its quartiles (four groups of equal size). Each quartile holds about 12,000 skills, and on average about 930 of them are copied, so the shares are stable.

Four rules, written in advance, turned these findings into preprocessing steps (Table 2). Constant columns carry no information and are removed. Counts that are very skewed — most skills have no links or code, while a few have thousands — would let a handful of extreme skills dominate linear and distance-based models, so columns with skewness above 2 are log-transformed for those models; the extreme values are kept, because they are real skills, not errors. Near-duplicate columns (Spearman correlation above 0.95) make the coefficients of linear models unstable, so those models keep only one column of each pair. Impossible values are treated as missing. We did not resample the data to correct the imbalance between copied and other skills: class weights do the same job inside the learning algorithm and were tuned as a hyperparameter. All steps are fitted inside each training fold, in scikit-learn pipelines \cite{pedregosa2011scikit}, so that no information from the validation data leaks into training.

| Step | Decision (and the finding that motivated it) |
|----------------------|--------------------------------------------------------------------|
| Missing values | Median imputation with indicator (numeric), constant category (categorical); only 0.02% missing |
| Categorical encoding | One-hot, categories with fewer than ten occurrences grouped; long tails of languages and licenses |
| Normalization | Standardization for scale-sensitive models, none for trees; very different scales |
| Outliers | Kept, with log transform for scale-sensitive models; 21 text columns with skewness above 2 (Figure 2a) |
| Discretization | Not used; trees choose their own cut points, and bins would discard information |
| Class imbalance | No resampling, class weight tuned, threshold-free main metric; 7.7% of skills copied |
| Dimensionality | One column of each near-duplicate pair removed for scale-sensitive models; eight pairs above 0.95 (Figure 2b) |

: Preprocessing decisions and the findings that motivated them.

## 3.6 Model development

All algorithms were trained and evaluated on the same development-set partitions using 3-fold cross-validation repeated three times with different seeds.
Folds are grouped by author to prevent author overlap and stratified to preserve the proportion of copied skills across folds.
Each configuration therefore yields nine estimates, reducing dependence on any single partition.

As an initial spot-check, we compared seven algorithms with default settings, covering linear (logistic regression), probabilistic (Gaussian Naive Bayes), instance-based (k-nearest neighbors), tree-based (decision tree), ensemble (random forest and HistGradientBoosting), and neural network (multilayer perceptron) approaches. This comparison identified promising algorithms for subsequent hyperparameter tuning.

The three best-performing algorithms were then tuned using nested cross-validation.
In each of six outer folds (the first two repetitions of the partitions above), random search sampled ten hyperparameter configurations (Table 3), evaluated each across three inner folds of the training data, and selected the best.
That configuration was then measured on the outer fold, which the search had never seen, so the estimate is not inflated by the choice.
Each algorithm was also evaluated using its default settings.
We selected the configuration with the highest mean AUC; when differences were not statistically significant, we preferred the fastest configuration.

| Algorithm | Hyperparameters searched (10 random configurations) |
|------------------------------|----------------------------------------------------------|
| HistGradientBoosting | learning rate 0.02–0.2 (log); 100–500 iterations; 8–64 leaves; 5–100 samples per leaf; L2 penalty 0.001–10 (log); class weight none or balanced |
| Logistic regression | inverse regularization C 0.001–100 (log); class weight none or balanced |
| Multilayer perceptron | hidden layers (64), (128) or (64, 32); L2 penalty $10^{-5}$–$10^{-2}$ (log); initial learning rate $10^{-4}$–$10^{-2}$ (log); early stopping |

: Search spaces of the three algorithms selected by the spot-check.

**Final models.** The selected configuration was trained on the whole development set with the text attributes. This **text model** answers RQ1 and is the one we interpret for RQ2. For comparison, the same configuration was trained with three other sets of attributes: **size only**, to see whether the text says more than its length; the **origin repository**; and the **origin repository plus the text**. We also fixed a decision threshold, needed only for metrics such as precision and recall, as the one that maximizes balanced accuracy on the development set. All models and the threshold were saved, with a fingerprint of each file, before the test set was opened.

## 3.7 Evaluation

Our primary metric is the area under the ROC curve (AUC), which measures how well the model ranks copied skills above non-copied ones. This metric captures whether text ranks skills by their likelihood of being copied, independently of the classification threshold or class prevalence. Given the rarity of copied skills, we also report PR AUC (relative to the proportion of copied skills), the Brier score (probability calibration), and, at the selected threshold, precision, recall, balanced accuracy, and the confusion matrix.

We assess statistical significance using fold-level comparisons on the development set.
Since fold training sets overlap, the nine results are not independent and an ordinary paired test would be too optimistic; the Nadeau–Bengio correction accounts for this overlap \cite{nadeau2003inference}, and Holm's method adjusts for several comparisons made at once \cite{holm1979simple}.
The test set is evaluated only once, with 95% confidence intervals estimated from 2,000 bootstrap samples drawn by author, since skills of the same author are not independent. 
RQ1 is supported if the text model's AUC confidence interval lies above 0.5 (random guess).
The control is passed if the interval of the gain from adding the text to the origin repository lies above zero.

Finally, we test robustness through three prespecified analyses: alternative time windows (30 or 90 days), aggregator definitions (500 skills instead of 100, or no exclusion, requiring dates for the 84,995 aggregator copies), and a temporal split (training on earlier skills and testing on later ones). The latter most closely approximates future prediction.

## 3.8 Interpretation

To answer RQ2, we interpret the text model on 20,000 test-set skills using two complementary methods:

- **Permutation importance** measures the AUC decrease after shuffling an attribute, indicating its contribution to model performance \cite{breiman2001statistical,molnar2020interpretable}. We also permute entire Table 1 attribute families to account for correlations.

- **SHAP values** quantify each attribute's contribution to individual predictions and how it varies with attribute values \cite{lundberg2017unified}. In logistic regression, these contributions correspond to the coefficient multiplied by the attribute's deviation from its mean.

Finally, we repeat permutation importance across the nine development folds to assess ranking stability.
