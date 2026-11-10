# 1. Introduction

## 1.1 Context and motivation

Agent skills are portable directories that combine structured metadata, natural-language instructions and, optionally, scripts and references; the `SKILL.md` file tells an LLM agent how to perform a task, which constraints to observe and which resources to use \cite{agentskills_spec}. Software reuse has been studied for decades \cite{haefliger2008code,lopes2017dejavu}, but skills have no central registry or package manager: reuse happens by copying folders between repositories. The same skill may appear in many repositories because it is clear and portable, because it belongs to a template or catalog, because a tool installed it, or because a single account duplicated it across its own projects. These explanations mix properties of the artifact and of the distribution context.

GitSkills provides a large-scale corpus of skill occurrences, distinct contents and repository metadata \cite{gitskills2027}. Other studies examine specification defects \cite{zhang2026keepskillsreuse}, the effect of skills on software engineering tasks \cite{han2026sweskillsbench} and the evolution of prompts \cite{tafreshipour2025prompting}. A complementary and methodologically delicate question remains: which observable characteristics are associated with the diffusion of a skill across distinct owners, and how much of this predictability results from how the dataset itself was built?

## 1.2 Research problem

We operationalize reuse as the presence of identical `SKILL.md` content (same `content_hash`) in distinct repositories. A first version of this study modeled the number of repositories (`repository_count`) from skill, text and representative-repository attributes, with repository-grouped validation and an external test set. Two audits showed that this design was fragile. First, GitSkills keeps detailed metadata for only one **representative** copy of each content, chosen by the best location class (preferring `.claude/skills/`) and then alphabetically; since a content with more copies is more likely to have some copy in the preferred folder, the representative's location class, and the commit-history indicator that depends on it, encode the target itself. Second, 28% of reused skills have all copies under a **single owner**, and one account alone accounts for 12% of the development set, so counting repositories mixes diffusion among developers with duplication by one account.

We therefore reformulated the problem. The main target is **cross-owner reuse** (`cross_owner`: the content appears under two or more owners) and, secondarily, the number of owners (`owner_count`); the attributes exclude the variables that depend on the representative rule, which are analyzed as a finding. The evaluation uses owner-grouped validation, a per-owner cap, nested tuning and pre-specified hypotheses, and confirmation is done on a **new holdout, disjoint by owner**, evaluated once. Because the study was revised after exploratory results, we distinguish what is confirmatory from what is exploratory. Conclusions are observational associations \cite{shmueli2010explain}: we claim neither causality nor skill quality.

## 1.3 Objectives and research questions

* **RQ1 — Construction biases:** to what extent do the representative-selection rule, self-copying and owner concentration confound the target and the attributes of GitSkills?
* **RQ2 — Predictability:** with attributes free of these artifacts, can cross-owner reuse and the number of owners be predicted better than baselines, and is the estimate confirmed on an owner-disjoint holdout?
* **RQ3 — Origin of the signal:** what is the contribution of skill, repository and catalog attributes, which variables are associated with the predictions, and in which direction?

## 1.4 Contributions

(i) An **audit of the GitSkills representative**: we recover its rule (97.2% agreement) and show that location and history depend on the number of copies. (ii) A **protocol** for cross-owner reuse with an owner-based target, a per-owner cap, grouped validation, pre-specified hypotheses and confirmation on a new holdout. (iii) Evidence of a **modest predictive signal**, with a stable RMSLE gain across protocol variations, decomposed into blocks and confirmed on unseen data (AUC 0.73). (iv) A **reproducible package** with the data revision, seeds, a five-minute reproduction path and a lock that prevents reusing the holdout.
