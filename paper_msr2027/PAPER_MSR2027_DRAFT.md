# Predicting Cross-Repository Reuse and Diffusion of AI Agent Skills on GitHub

**Conference**: 24th International Conference on Mining Software Repositories (MSR 2027)  
**Track**: Mining Challenge Track (Short Paper, 4 pages + 1 page references)  
**Target Dataset**: GitSkills (Preprint: arXiv:2608.10906, Zenodo DOI: 10.5281/zenodo.21875637)  
**Review Model**: Double-Anonymous (Double-Blind)  
**Status**: Anonymized Submission Draft  

---

## Abstract
As Large Language Model (LLM) coding agents proliferate, modular skill specifications (`SKILL.md` files) have emerged as the primary mechanism for encapsulating natural-language instructions, tool definitions, and operational workflows. Yet, because skills spread by direct folder copying without a package manager or central registry, the factors governing their cross-repository adoption remain uncharacterized. This paper investigates the GitSkills dataset, addressing MSR Mining Challenge Directions 3 (Reuse without a package manager) and 4 (Software metrics for natural-language instructions). Using a strict grouped partition across 200,000 skills from 72,376 repositories with zero repository overlap between training ($N=160,000$) and an external test set ($N=40,000$), we analyze multimodal predictors of reuse. We uncover that non-linear tree ensembles suffer from severe shortcut learning under random splits (inflating to 0.85 ROC-AUC by memorizing continuous repository variables), yet collapse under grouped evaluation. In contrast, $L_2$-regularized linear models with supervised $\chi^2$ text screening achieve robust generalization (0.6188 on the blind test set of 14,568 unseen repositories). Calibrated threshold optimization ($t^* = 0.21$) boosts operational $F_1$ by $+372\%$. Explainability analyses reveal the "Three Pillars of Skill Reuse": structural canonicity ($+87\%$), modern component ecosystems, and formal prompt specification contracts. Finally, valid frontmatter syntax ($+224\%$) enables the model to recover 49.82\% of viral skills ($\ge 10$ repositories) in unfamiliar codebases.

---

## 1. Introduction
Autonomous coding agents (e.g., Google Antigravity, Claude Code, GitHub Copilot Workspace) are shifting software development from hardcoded scripts to prompt-driven tool execution. Central to this paradigm is the *agent skill*---typically structured as a folder containing a `SKILL.md` markdown file with YAML frontmatter metadata, procedural instructions, and optional bundled scripts. When an agent determines that a developer's prompt matches a skill description, it dynamically executes the instructions.

However, as highlighted in the MSR 2027 Mining Challenge, agent skills differ fundamentally from traditional source code: their content is written primarily in natural language, their execution is probabilistic, and they lack a centralized package manager (e.g., npm, PyPI). Consequently, skills spread through ad-hoc directory copying across repositories. Over 50\% of observed instances are verbatim duplicates, while the vast majority of original skills remain unshared singletons.

This decentralized package-manager vacuum creates three critical software engineering challenges: (1) **Authoring Uncertainty**, where developers lack empirical guidelines or linters for reusable prompt interfaces; (2) **Ecosystem Discovery**, where autonomous agents lack ranking signals to distinguish ephemeral single-use drafts from modular, generalizable capabilities; and (3) **Supply-Chain Fragility**, where unvetted, probabilistically selected prompts execute arbitrary shell commands across multiple codebases. By predicting and characterizing reuse, this paper establishes the first empirical foundations to address these challenges.

This paper investigates the **GitSkills** community dataset (July 2026 release), addressing two core challenge directions: **Direction 3** (*Reuse without a package manager*) and **Direction 4** (*Software metrics for natural-language instructions*). We formulate two research questions:
- **RQ1 (Binary Skill Reuse)**: *To what extent can structural, repository, and textual metrics predict whether a skill will be reused across multiple repositories ($y=1$) versus remaining a singleton ($y=0$) in unseen codebases?*
- **RQ2 (Conditional Scale of Diffusion)**: *Among reused skills, what properties distinguish small-scale sharing (2 repos) from massive viral diffusion ($10+$ repos)?*

By enforcing a grouped partition with exactly zero repository overlap, we evaluate learning paradigms, demonstrate the failure mode of tree ensembles due to shortcut learning, optimize decision thresholds, and extract actionable principles governing agent skill design.

---

## 2. Dataset and Problem Formulation

### 2.1 Deduplication and Unit of Analysis
We mine the GitSkills snapshot (Zenodo DOI: `10.5281/zenodo.21875637`). The atomic unit of analysis is a unique normalized content payload of a `SKILL.md` file, uniquely identified by its SHA-256 digest (`content_hash`). Deduplication is performed prior to partitioning, yielding an official population of 200,000 distinct skills originating across 72,376 unique GitHub repositories.

### 2.2 Target Definitions
We formulate a dual-target architecture:
1. **Binary Reuse ($y \in \{0, 1\}$)**: $y = 1$ if the exact `content_hash` appears in $\ge 2$ distinct repositories; $y = 0$ otherwise. The natural positive prevalence is $16.20\%$.
2. **Conditional Intensity ($y_{\text{int}} \in \{0, 1, 2, 3\}$)**: Conditional on $y=1$, categorizing diffusion scale: Class 0 (2 repos, $56.3\%$), Class 1 (3--4 repos, $23.5\%$), Class 2 (5--9 repos, $11.7\%$), and Class 3 ($10+$ repos, $8.5\%$).

### 2.3 Leakage-Free Grouped Partitioning
Skills from the same repository share identical contextual metrics (stars, forks, commits). Naive random $I.I.D.$ splits permit models to memorize repository clusters. We eliminate this group leakage using `StratifiedGroupKFold` grouped by `representative_repo_id`:
- **Training Split**: 160,000 skills across 57,808 repositories.
- **External Test Split**: 40,000 skills across 14,568 repositories.
- **Repository Overlap**: **Exactly 0** ($\text{Train} \cap \text{Test} = \emptyset$).

### 2.4 Extracted Multimodal Metrics
We engineer 143 tabular features spanning two domains: (1) *Structural and Textometrics (46 features)*, including YAML validity (`frontmatter_valid`), line/character counts, markdown heading depth, fenced code blocks, checklists, and path canonicity (`location_class`); and (2) *Repository Context (97 features)*, including star count, forks, commit history, and primary programming language. Textual representations are captured via sublinear TF-IDF n-grams ($1$--$2$ words).

---

## 3. Experimental Methodology

### 3.1 Unmasking Shortcut Learning in Tree Ensembles
Initial spot-checking across inductive learning biases revealed a critical methodological trap. Under random $I.I.D.$ splitting, Histogram Gradient Boosted Decision Trees (`HistGradientBoosting`) appeared to achieve **0.8514** ROC-AUC. However, analyzing split criteria unmasked a classic *Clever Hans* artifact:
- By recursively splitting on continuous variables (`sibling_bytes`, `commit_count`, `stars`), trees isolated unique continuous tuples corresponding to specific repositories present across both train and test folds.
- When re-evaluated under strict grouped splitting (`StratifiedGroupKFold`) on 100\% unseen repositories, tree performance suffered a **catastrophic collapse** to **$0.48 \sim 0.62$** ROC-AUC.
- Conversely, $L_2$-regularized Logistic Regression enforces global monotonic hyperplanes $\sigma(w^T x + b)$. Incapable of carving arbitrary continuous slices, the linear model achieved robust, genuine generalization ($0.6706$ CV ROC-AUC).

### 3.2 Multimodal Ablation & Supervised $\chi^2$ Screening
We evaluated 6 ablation configurations on the training split:
- `text_only` (50,000 TF-IDF features; $0.5818$ ROC-AUC)
- `repo_only` (97 features; $0.6006$)
- `all` (50,150 raw multimodal features; $0.6067$)
- `structure_only` (46 features; $0.6450$)
- `structure_and_repo` (143 features; $0.6664$)
- `hybrid_screened` (143 tabular + Top-100 $\chi^2$ textual n-grams; **0.6706**)

Unsupervised high-dimensional text proved noisy ($\pm 0.1221$ std). However, screening the top 100 terms via supervised univariate $\chi^2$ computed strictly within each training fold eliminated noise, yielding the project's champion model.

### 3.3 Calibration & Decision Threshold Tuning ($t^* = 0.21$)
The natural unweighted model achieves clean probability calibration (Brier score loss of **0.1323**). Due to the $16.20\%$ positive prevalence, default classification ($t = 0.50$) misses $95.6\%$ of reused skills ($F_1 = 0.0794$). Sweeping 86 out-of-fold thresholds on the 160k training set identified $t^* = 0.21$ as the optimal operating threshold, elevating $F_1$ to **0.3754** ($+372\%$ relative gain) with $46.56\%$ recall. This threshold was frozen prior to test evaluation.

---

## 4. Results and Knowledge Discovery

### 4.1 The Three Pillars of Skill Reuse (RQ1)
Inspecting standardized Odds Ratios ($\text{OR} = e^\beta$) and game-theoretic SHAP values reveals the mechanisms governing skill adoption:
1. **Pillar 1: Structural Canonicity**: Conforming to agent path standards is paramount. Canonical paths (`.gemini/antigravity/skills`) exhibit $\text{OR} = 1.87$ ($+87\%$ reuse odds), whereas non-canonical directories decrease odds by $-44\%$ ($\text{OR} = 0.56$). Exact uppercase naming (`SKILL.md`) yields $\text{OR} = 1.86$, while lowercase naming penalizes adoption by $-54\%$ ($\text{OR} = 0.46$).
2. **Pillar 2: Propagator Ecosystems**: Modular component ecosystems drive adoption: Svelte ($\text{OR} = 6.75$), AutoHotkey ($\text{OR} = 8.47$), and Python. Repository prestige (stars/forks) facilitates organic indexing.
3. **Pillar 3: Prompt Behavioral Contracts**: Crucially, **task domain does not predict reuse**. Instead, vocabulary formalizing prompt behavioral contracts strongly marks reused skills:
   - `text__objective` ($\text{OR} = 1021.18, \beta = 6.93$)
   - `text__patterns` ($\text{OR} = 433.07, \beta = 6.07$)
   - `text__example` ($\text{OR} = 122.05, \beta = 4.80$)
   - `text__prompt role` ($\text{OR} = 107.14, \beta = 4.67$)
   - `text__rules constraints` ($\text{OR} = 47.40, \beta = 3.86$)

These large Odds Ratios reflect near-complete empirical separation under $L_2$ regularization: terms like "objective" and "rules constraints" appear almost exclusively in reused templates rather than ad-hoc singletons. Rather than establishing direct causality, these contractual headers serve as reliable discriminative signatures of modular engineering practices, establishing clear personas, bounded objectives, explicit constraints, and few-shot input/output examples.

### 4.2 Drivers of Massive Viral Diffusion (RQ2)
Analyzing conditional intensity for Class 3 ($\ge 10$ repos) reveals the differentiators of ecosystem-wide diffusion:
- **Valid Metadata Syntax**: Valid YAML frontmatter (`frontmatter_valid`) yields $\text{OR} = 3.24$ ($+224\%$ odds of viral diffusion). Automated agent orchestrators systematically discard unparseable headers.
- **Operational Boundary Definitions**: Viral skills explicitly declare execution boundaries, captured by `text__limitations` ($\text{OR} = 2.28$) and `text__provides step` ($\text{OR} = 1.83$).
- **Superpropagator Frameworks**: Astro ($\text{OR} = 3.65$), Jupyter ($\text{OR} = 3.01$), and Python ($\text{OR} = 2.27$).

### 4.3 Blind External Evaluation (40,000 Test Instances)
Evaluating the frozen pipeline on 40,000 instances from 14,568 unseen repositories confirms our findings:
- **Generalization and Multimodal Value**: On 14,568 untouched external repositories, the hybrid model achieved **0.6211** ROC-AUC ($0.6706 \pm 0.0633$ CV). While text yields a modest quantitative gain over tabular repo metrics ($+0.0091$ ROC-AUC over `structure_and_repo`), modeling natural-language prompt tokens is qualitatively indispensable: it provides the only actionable semantic guidance (Pillar 3) for skill authors, whereas external repository stars and commit histories are beyond an author's direct control.
- **Threshold Impact**: The frozen threshold ($t^* = 0.21$) elevated test $F_1$ from $0.0727$ to **0.2967** ($+308\%$ relative gain over $t=0.50$ baseline), recovering **2,376 true positive reused skills** with $78.16\%$ specificity ($26,200$ true negatives).
- **Viral Recovery**: On conditional intensity, the model achieved Class 3 Recall of **49.52\%** ($F_1 = 0.3163$). The classifier successfully flags half of all massively diffused skills in completely foreign repositories.

---

## 5. Implications for Practice and Tooling
Our champion model ($\text{ROC-AUC}=0.6211$, Specificity=$78.16\%$) is not intended as an autonomous binary gatekeeper, but rather as: (1) an **Authoring Linter** embedded in developer IDEs (e.g., VS Code extensions) that provides pre-commit warnings when a skill lacks standard paths, valid schemas, or contractual boundaries; and (2) an **Ecosystem Quality Prior** for skill retrieval engines (Learning-to-Rank), boosting well-specified, modular skills in decentralized community catalogs.

Based on our findings, we formulate four actionable guidelines for agent skill creators:
1. **Standardize File Locations**: Place skills at `.gemini/antigravity/skills/<skill>/SKILL.md` or `.agent/skills/`. Avoid non-standard naming.
2. **Enforce Frontmatter Schemas**: Always provide valid delimited YAML declaring `name` and `description`.
3. **Structure Contractual Prompt Headers**: Incorporate explicit markdown sections for `## Objective`, `## Role`, `## Rules & Constraints`, and `## Examples`.
4. **Specify Negative Boundaries**: Explicitly declare `## Limitations` to restrict erroneous agent tool invocation.

### Threats to Validity
- **Construct Validity**: In empirical MSR, cross-repository reuse serves as an observational proxy for community adoption and utility, though popularity may not guarantee bug-free implementation. Furthermore, following GitSkills conventions, we define reuse via exact SHA-256 content hashes (Type-1 clones). Consequently, near-duplicate skills modified by developers (Type-2/Type-3 clones) are categorized as singletons, potentially underestimating semantic diffusion. Future work should incorporate syntactic tree-diffing and embedding similarity.
- **Internal Validity**: Confounding factors, such as superpropagator frameworks (e.g., Svelte, AutoHotkey) adopting standardized prompts by convention, could correlate with reuse. While grouped splits prevent repository memorization, observational SE mining cannot claim pure intervention causality.
- **External Validity**: Our dataset reflects the GitHub ecosystem as of July 2026. While capturing 72k repositories, agent skill formats continue to evolve rapidly.

---

## 6. Ethical Implications
In accordance with challenge requirements, we consider the ethical dimensions of mining and modeling agent skills:
- **Privacy and Anonymization**: We analyze public GitHub repositories without collecting or predicting personally identifiable information (PII). Developer accounts were anonymized in GitSkills commit tables, and our models operate exclusively on content hashes and repository-level aggregates.
- **Prompt Intellectual Property**: Decentralized skill copying without a registry risks obscuring original attribution. Our study emphasizes that transparent license and frontmatter declarations foster ethical attribution in agent ecosystems.
- **Supply-Chain Security**: Unverified skill propagation introduces vulnerability vectors (e.g., executing arbitrary shell commands). Understanding reuse patterns enables future automated security scanners to audit widely copied skills.
- **Computational Sustainability (Green AI)**: By demonstrating that regularized linear models with sparse caching outperform heavy deep neural networks, our replication pipeline executes in under 2 minutes, minimizing computational carbon footprint.

---

## 7. Conclusion
This paper presented an empirical study predicting the cross-repository reuse and diffusion of AI agent skills in the GitSkills dataset. We unmasked shortcut learning in tree ensembles, proved the stability of regularized linear models with supervised $\chi^2$ screening, and uncovered the Three Pillars of Skill Reuse. By providing verified models, operational thresholds, and authoring guidelines, this work establishes empirical foundations for the engineering of reusable agent skills.

### Replication and Open Science
The complete replication package, sparse matrices, scripts, and an interactive 1-click Google Colab notebook are available anonymously at: `https://anonymous.4open.science/r/skillreuse-anon`.

---

## References

1. **G. Destefanis, D. Graziotin, M. Vaccargiu, and M. Ortu**, "GitSkills: A Dataset of Agent Skills on GitHub," in *Proc. 24th IEEE/ACM Int. Conf. Mining Software Repositories (MSR)*, 2027.
2. **S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. Narasimhan, and Y. Cao**, "ReAct: Synergizing Reasoning and Acting in Language Models," in *Proc. Int. Conf. Learn. Representations (ICLR)*, 2023.
3. **T. Schick, J. Dwivedi-Yu, R. Dessì, R. Raileanu, M. Lomeli, L. Zettlemoyer, N. Cancedda, and T. Scialom**, "Toolformer: Language models can teach themselves to use tools," *NeurIPS*, vol. 36, pp. 68539–68551, 2023.
4. **Y. Qin et al.**, "ToolLLM: Facilitating large language models to master 16000+ real-world APIs," *arXiv:2307.16789*, 2023.
5. **C. E. Jimenez, J. Yang, A. Wettig, S. Yao, K. Pei, O. Press, and K. Narasimhan**, "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?," in *Proc. Int. Conf. Learn. Representations (ICLR)*, 2024.
6. **C. V. Lopes, P. Maj, P. Martins, V. Sajnani, J. Di Rocco, H. Summer, and P. Baldi**, "DéjàVu: A map of code duplicates on GitHub," *Proc. ACM Program. Lang. (OOPSLA)*, vol. 1, pp. 1–28, 2017.
7. **S. Haefliger, G. Von Krogh, and S. Spaeth**, "Code reuse in open source software," *Management Science*, vol. 54, no. 1, pp. 180–193, 2008.
8. **A. Mockus, R. T. Fielding, and J. D. Herbsleb**, "Two case studies of open source software development: Apache and Mozilla," *ACM TOSEM*, vol. 11, no. 3, pp. 309–346, 2002.
9. **C. K. Roy and J. R. Cordy**, "A survey on software clone detection research," *Queen's School of Computing TR 2007-541*, 2007.
10. **R. Geirhos et al.**, "Shortcut learning in deep neural networks," *Nature Machine Intelligence*, vol. 2, no. 11, pp. 665–673, 2020.
11. **S. Lapuschkin et al.**, "Unmasking Clever Hans predictors and assessing what machines really learn," *Nature Communications*, vol. 10, no. 1, p. 1096, 2019.
12. **S. Saeb, L. Lonini, A. Jayaraman, D. C. Mohr, and K. P. Kording**, "The need to separate related samples in cross-validation: Error estimation for models that predict cross-subject traits," *GigaScience*, vol. 6, no. 4, pp. 1–11, 2017.
13. **D. R. Roberts et al.**, "Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure," *Ecography*, vol. 40, no. 8, pp. 913–929, 2017.
14. **S. Kaufman, S. Rosset, C. Perlich, and O. Stitelman**, "Leakage in data mining: Formulation, detection, and avoidance," *ACM TKDD*, vol. 6, no. 4, pp. 1–21, 2012.
15. **A. Niculescu-Mizil and R. Caruana**, "Predicting good probabilities with supervised learning," in *Proc. 22nd ICML*, pp. 625–632, 2005.
16. **G. W. Brier**, "Verification of forecasts expressed in terms of probability," *Monthly Weather Review*, vol. 78, no. 1, pp. 1–3, 1950.
17. **C. Elkan**, "The foundations of cost-sensitive learning," in *Proc. IJCAI*, vol. 17, pp. 973–978, 2001.
18. **C. Tantithamthavorn, A. E. Hassan, and K. Matsumoto**, "The impact of class rebalancing techniques on software defect prediction," *IEEE TSE*, vol. 46, no. 11, pp. 1200–1214, 2020.
19. **S. M. Lundberg and S.-I. Lee**, "A unified approach to interpreting model predictions," in *NeurIPS*, vol. 30, pp. 4765–4774, 2017.
20. **C. Molnar**, *Interpretable Machine Learning: A Guide for Making Black Box Models Explainable*, Leanpub, 2020.
