# Predicting Cross-Repository Reuse and Diffusion of AI Agent Skills: A Multimodal Empirical Study on GitHub

**Author**: Pedro Odorize  
**Affiliation**: Instituto de Informática – Universidade Federal do Rio Grande do Sul (UFRGS)  
**Email**: `pedro.odorize@inf.ufrgs.br`  
**Course**: CMP263 - Machine Learning (Profa. Mariana Recamonde Mendoza)  

---

## Abstract
As Large Language Model (LLM) agents proliferate, modular skill specifications (such as `SKILL.md` files) have emerged as the standard mechanism for encapsulating tool invocations, system instructions, and operational workflows. However, the factors driving cross-repository reuse and wide diffusion of such skills remain largely unknown. This paper presents an empirical study on 200,000 unique agent skills across 72,376 GitHub repositories. Using a strict grouped partition that guarantees zero repository overlap between training ($N=160,000$) and an external blind test set ($N=40,000$), we evaluate multiple learning paradigms and perform systematic multimodal ablations. We discover that non-linear tree ensembles suffer from severe shortcut learning under random splits (inflating to 0.85 ROC-AUC by memorizing continuous repository variables), yet catastrophically collapse to near-random performance under grouped evaluation. Conversely, $L_2$-regularized linear models combined with supervised $\chi^2$ text screening achieve robust generalization (0.6706 CV ROC-AUC; 0.6188 on the blind test set of 14,568 unseen repositories). Probability calibration and decision threshold tuning ($t^*=0.21$) improve $F_1$-score by $+372\%$. Explainability analyses via Odds Ratios and SHAP reveal the "Three Pillars of Skill Reuse": structural canonicity ($+87\%$), modern software ecosystems, and formal prompt specification contracts. Furthermore, valid frontmatter syntax ($+224\%$) and operational boundary definitions enable the model to recover 49.82\% of viral, massively diffused skills ($\ge 10$ repositories) in entirely unfamiliar codebases.

## Resumo
Com a proliferação de agentes baseados em LLMs, especificações modulares de habilidades (`SKILL.md`) tornaram-se o mecanismo padrão de encapsulamento de ferramentas e fluxos operacionais. Contudo, os determinantes de seu reúso entre repositórios permanecem desconhecidos. Este artigo apresenta um estudo empírico sobre 200.000 habilidades únicas em 72.376 repositórios do GitHub. Empregando partição agrupada rígida com sobreposição nula entre treino ($N=160.000$) e teste cego externo ($N=40.000$), avaliamos diferentes paradigmas indutivos e ablações multimodais. Modelos baseados em árvores sofrem aprendizado de atalho sob divisão aleatória (inflacionando para 0,85 de ROC-AUC ao memorizar repositórios), mas colapsam sob partição agrupada. Em contrapartida, modelos lineares regularizados com triagem $\chi^2$ atingem generalização robusta (0,6188 no teste cego com 14.568 repositórios inéditos). Calibração e ajuste de limiar ($t^*=0,21$) elevam o $F_1$ em $+372\%$. Análises por Odds Ratios e SHAP revelam os Três Pilares do Reúso: canonicidade estrutural ($+87\%$), ecossistemas tecnológicos e contratos formais de prompt, além de recuperar 49,82\% dos superpropagadores massivos ($\ge 10$ repositórios).

---

## 1. Introduction

The rapid transition from conversational Large Language Models (LLMs) to autonomous agentic architectures has fundamentally altered software development. Rather than executing monolithic, hardcoded automation scripts, modern software engineering agents dynamically discover, parse, and execute specialized capabilities known as **agent skills**. Popularized by developer environments such as Google Antigravity, Claude Code, and GitHub Copilot Workspace, these skills are conventionally declared in modular Markdown specifications titled `SKILL.md`. These files encapsulate natural language operational contracts, executable tool declarations, parameter constraints, and contextual domain demonstrations.

As thousands of open-source projects adopt this paradigm, developers frequently copy, adapt, and import third-party skills to enhance their autonomous coding workflows. In open-source software engineering, component reuse has historically served as a primary catalyst for developer productivity, software quality, and architectural convergence. However, while decades of Mining Software Repositories (MSR) research have examined source code duplication, package dependency networks, and API reuse, the reuse dynamics of **natural language prompt artifacts** and **agent skill specifications** remain entirely uncharacterized.

Predicting and understanding skill reuse presents unique conceptual and empirical challenges:
1. **Multimodal Heterogeneity**: Skills combine free-form instructional text, structured YAML frontmatter, textometric code metrics, and broader repository context (e.g., star count, language, and commit history).
2. **Extreme Grouped Correlation and Data Leakage**: Skills originating within the same GitHub repository share identical contextual metadata. Standard randomized train/test splitting induces insidious group leakage, causing non-linear algorithms to memorize repository identities via continuous variables.
3. **Severe Class Imbalance and Diffusion Asymmetry**: Over 83\% of skills exist as isolated singletons in only one repository, whereas a minute fraction diffuses virally across tens or hundreds of distinct projects.

### Research Questions
- **RQ1 (Binary Skill Reuse)**: *To what extent can structural, repository, and textual attributes distinguish reusable `SKILL.md` contents (present in $\ge 2$ repositories) from singletons ($= 1$ repository) in previously unseen repositories?*
- **RQ2 (Conditional Intensity of Diffusion)**: *Among skills that are successfully reused, what properties govern their scale of diffusion across the open-source ecosystem (from small-scale sharing of 2 repos to viral diffusion across $10+$ repos)?*

---

## 2. Dataset and Problem Formulation

### 2.1 Deduplicated Population and Unit of Analysis
The atomic unit of analysis is a unique normalized content payload of a `SKILL.md` file, uniquely identified by its cryptographic SHA-256 digest (`content_hash`). Content deduplication is performed strictly prior to dataset splitting, ensuring that identical copies across different forks or directories do not inflate sample counts artificially.

### 2.2 Dual-Target Formulation
1. **Binary Reuse ($y \in \{0, 1\}$)**: $y = 1$ if the exact `content_hash` is observed in $\ge 2$ distinct GitHub repositories, and $y = 0$ if it is a singleton observed in exactly one repository.
2. **Conditional Intensity ($y_{\text{int}} \in \{0, 1, 2, 3\}$)**: Applicable strictly conditional on $y = 1$, representing the scale of ecosystem diffusion:
   - Class 0 ($2$ repositories): Small-scale paired sharing ($56.3\%$ of reused skills).
   - Class 1 ($3$--$4$ repositories): Moderate diffusion ($23.5\%$).
   - Class 2 ($5$--$9$ repositories): Broad adoption ($11.7\%$).
   - Class 3 ($10+$ repositories): Massive viral diffusion ($8.5\%$).

### 2.3 Leakage-Free Grouped Stratified Partitioning
To eliminate data leakage, we partition the dataset using `StratifiedGroupKFold`:
- **Training Partition**: $160,000$ unique skills across $57,808$ repositories.
- **External Test Partition**: $40,000$ unique skills across $14,568$ repositories.
- **Repository Overlap**: **Exactly 0** ($\text{Train} \cap \text{Test} = \emptyset$).
- **Target Prevalence**: Preserved at exactly $16.20\%$ in both training ($25,918 / 160,000$) and test ($6,479 / 40,000$).

---

## 3. Experimental Methodology

### 3.1 Inductive Bias Spot-Checking & Shortcut Learning
Under randomized $I.I.D.$ cross-validation, Histogram Gradient Boosting achieved an extraordinary ROC-AUC of **0.8514**. However, inspecting tree splits revealed an immediate pathology:
- Multiple skills from the same repository shared identical continuous values for `sibling_bytes`, `commit_count`, and `repo_stars`. Decision trees recursively partitioned continuous feature space to isolate individual repository IDs present in both training and test folds.
- Under strict grouped cross-validation (`StratifiedGroupKFold`) on 100\% unseen repositories, tree ensembles suffered **catastrophic generalization collapse**, dropping to **$0.48 \sim 0.62$** ROC-AUC.
- Conversely, $L_2$-regularized Logistic Regression restricts hypothesis complexity to a global monotonic hyperplane $\sigma(w^T x + b)$. Unable to isolate arbitrary continuous feature slices, the linear model achieved robust, stable out-of-fold generalization (**$0.6706$** ROC-AUC).

### 3.2 Multimodal Ablation Suite
We evaluated 6 configurations under 3-Fold `StratifiedGroupKFold`:
1. `text_only` (50,000 TF-IDF features)
2. `repo_only` (~97 repository features)
3. `all` (50,150 raw multimodal features)
4. `structure_only` (46 structural features)
5. `structure_and_repo` (143 structural + repo features)
6. `hybrid_screened` (143 tabular features + Top-100 textual n-grams screened via supervised univariate $\chi^2$)

### 3.3 Probability Calibration & Threshold Optimization
The unweighted model achieved a Brier Score Loss of **0.1323**. Sweeping 86 out-of-fold decision thresholds ($t \in [0.05, 0.90]$) on the 160,000 training set identified the optimal threshold:
$$t^* = 0.21$$
This threshold boosted the cross-validation $F_1$-score from $0.0794$ (at 0.50) to **$0.3754$** (+372% relative increase) with Balanced Accuracy of **$63.47\%$**, and was strictly frozen for test evaluation.

---

## 4. Empirical Results and Interpretability

### 4.1 Cross-Validation Ablation Performance

| Configuration | Feature Space | ROC-AUC | PR-AUC | Intensity Macro $F_1$ |
| :--- | :--- | :--- | :--- | :--- |
| `text_only` | $50,000$ TF-IDF | $0.5818 \pm 0.1086$ | $0.2230 \pm 0.0422$ | $0.2505 \pm 0.0384$ |
| `repo_only` | $\sim 97$ Repo | $0.6006 \pm 0.0673$ | $0.2503 \pm 0.0538$ | $0.2584 \pm 0.0289$ |
| `all` (Raw Multimodal) | $50,150$ All | $0.6067 \pm 0.1221$ | $0.2662 \pm 0.0734$ | $0.2458 \pm 0.0401$ |
| `structure_only` | $46$ Structure | $0.6450 \pm 0.0142$ | $0.2372 \pm 0.0232$ | $0.2687 \pm 0.0354$ |
| `structure_and_repo` | $143$ Tabular | $0.6664 \pm 0.0511$ | $0.2721 \pm 0.0256$ | $0.2974 \pm 0.0366$ |
| **`hybrid_screened`** | **$143$ Tab + $100$ $\chi^2$** | **0.6706 $\pm$ 0.0633** | **0.2808 $\pm$ 0.0267** | **0.3110 $\pm$ 0.0750** |

### 4.2 The Three Pillars of Skill Reuse (Explainability Findings)
1. **Structural Canonicity**:
   - `location_class_canonical` (.gemini/antigravity/skills): $\text{OR} = 1.87$ ($+87\%$ reuse odds).
   - `filename_SKILL.md` (uppercase): $\text{OR} = 1.86$ ($+86\%$ reuse odds).
   - Non-canonical paths decrease odds by $-44\%$ ($\text{OR} = 0.56$), and lowercase naming decreases odds by $-54\%$ ($\text{OR} = 0.46$).
2. **Propagator Ecosystems**:
   - Modern modular frameworks lead adoption: `Svelte` ($\text{OR} = 6.75$), `AutoHotkey` ($\text{OR} = 8.47$), and `Python`.
3. **Formal Prompt Specification Contracts**:
   - Terms defining behavioral interfaces dominate:
     - `text__objective`: $\text{OR} = 1021.18$
     - `text__patterns`: $\text{OR} = 433.07$
     - `text__example`: $\text{OR} = 122.05$
     - `text__prompt role`: $\text{OR} = 107.14$
     - `text__best practices`: $\text{OR} = 88.41$
     - `text__rules constraints`: $\text{OR} = 47.40$

### 4.3 Drivers of Massive Diffusion (Class 3: $\ge 10$ Repositories)
- **Valid Frontmatter**: `frontmatter_valid` yields $\text{OR} = 3.24$ ($+224\%$ odds of viral adoption).
- **Operational Boundary Definitions**: Terms such as `text__limitations` ($\text{OR} = 2.28$) and `text__provides step` ($\text{OR} = 1.83$).
- **Superpropagator Frameworks**: Astro ($\text{OR} = 3.65$), Jupyter ($\text{OR} = 3.01$), and Python ($\text{OR} = 2.27$).

---

## 5. Final Blind External Evaluation (40,000 Instances)

Evaluating on 40,000 instances from 14,568 unseen repositories under the frozen pipeline:

| Model | Weighting | Threshold | ROC-AUC | PR-AUC | $F_1$-Score | Recall | Balanced Acc |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `structure_only` | Unweighted | $0.50$ | $0.6035$ | $0.2210$ | $0.0154$ | $0.79\%$ | $0.5025$ |
| `structure_only` | Unweighted | **$t^*=0.21$** | $0.6035$ | $0.2210$ | **$0.2667$** | $29.97\%$ | $0.5583$ |
| `structure_and_repo` | Unweighted | $0.50$ | $0.6120$ | $0.2033$ | $0.0565$ | $3.86\%$ | $0.4878$ |
| `structure_and_repo` | Unweighted | **$t^*=0.21$** | $0.6120$ | $0.2033$ | **$0.2887$** | $35.93\%$ | $0.5705$ |
| **`hybrid_screened`** | Unweighted | $0.50$ | $0.6188$ | $0.2083$ | $0.0611$ | $3.87\%$ | $0.4972$ |
| **`hybrid_screened`** | Unweighted | **$t^*=0.21$** | **$0.6188$** | **$0.2083$** | **$0.2884$** | **$35.89\%$** | **$0.5702$** |
| `hybrid_screened` | Balanced | $0.50$ | **$0.6213$** | $0.2085$ | **$0.3117$** | **$50.32\%$** | **$0.5848$** |

### Test Confusion Matrix ($t^* = 0.21$, $N = 40,000$)
- **True Negatives**: 26,200 (Specificity: $78.16\%$)
- **False Positives**: 7,321
- **False Negatives**: 4,154
- **True Positives**: **2,325** (Sensitivity/Recall: $35.89\%$)

### Viral Skill Recovery (Intensity Class 3)
On the conditional intensity test split ($N = 6,479$ reused skills), the hybrid model attained Class 3 $F_1$ of **0.3216** with a Recall of **49.82\%**, successfully recovering half of all viral skills across foreign repositories.

---

## 6. Actionable Guidelines for AI Skill Developers

1. **Adhere to Canonical Directory Hierarchy**: Store files at `.gemini/antigravity/skills/<skill-name>/SKILL.md` with uppercase naming.
2. **Validate YAML Frontmatter**: Include delimited YAML (`---`) with valid `name` and `description` fields.
3. **Structure Prompt Behavioral Contracts**: Declare explicit markdown sections for `Objective`, `Prompt Role`, `Rules & Constraints`, `Patterns`, and `Examples`.
4. **Document Operational Boundaries**: Explicitly catalog negative triggers and limitations (`limitations`).
