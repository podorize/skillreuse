# 1. Introduction

## 1.1 Context and motivation

The adoption of agents based on Large Language Models (LLMs) in software engineering has been accompanied by new artifacts designed to encapsulate operational knowledge. Among them are *agent skills*: portable directories that combine structured metadata, natural-language instructions and, optionally, scripts and references. The `SKILL.md` file is the descriptive core of the artifact and tells the agent how to perform a task, which constraints to observe and which resources to use \cite{agentskills_spec}. Skills extend the traditional notion of reuse, since they distribute procedures and domain knowledge without depending on a library with a formal interface.

Software reuse has been studied for decades \cite{haefliger2008code,lopes2017dejavu}, but the diffusion of agent skills has its own characteristics: there is no central registry or package manager, and reuse happens by copying folders between repositories. The same skill may appear in many repositories because it is clear and portable, because it belongs to a template or catalog, because a tool installed it, or because a single account duplicated it across its own projects. These explanations mix properties of the artifact and of the distribution context, which makes the observed reuse hard to interpret.

GitSkills provides a large-scale corpus with skill occurrences, distinct contents and repository metadata \cite{gitskills2027}. Other studies examine specification defects \cite{zhang2026keepskillsreuse}, the effect of skills on software engineering tasks \cite{han2026sweskillsbench} and the evolution of prompts \cite{tafreshipour2025prompting}. A complementary and methodologically delicate question remains: which observable characteristics are associated with the diffusion of a skill across distinct owners, and how much of this predictability results from how the dataset itself was built?

## 1.2 Research problem

We operationalize reuse as the presence of identical `SKILL.md` content (same `content_hash`) in distinct repositories. A first version of this study modeled the number of repositories (`repository_count`) from skill attributes, text and attributes of the representative repository, with repository-grouped validation and an external test set. Two audits, described in this report, showed that this design was fragile.

The first concerns the **representative**. GitSkills keeps detailed metadata for only one copy of each content, chosen by the best location class (preferring `.claude/skills/`) and then by the alphabetical order of repository and path. Because a content with more copies has more chances of having some copy in the preferred folder, the location class of the representative, and the commit-history indicator that depends on it, end up encoding the target itself. The second concerns **self-copying and concentration**: 28% of reused skills have all copies under a single owner, and one account alone accounts for 12% of the development set. Counting repositories mixes diffusion among developers with duplication by a single account.

We therefore reformulated the problem. The main target becomes **cross-owner reuse** (`cross_owner`: the content appears under two or more distinct owners) and, secondarily, the number of owners (`owner_count`). The attributes now exclude the variables that depend on the representative rule, which are analyzed as a finding. The evaluation uses owner-grouped validation, a per-owner cap, nested tuning and pre-specified hypotheses, and confirmation is done on a **new holdout, disjoint by owner**, evaluated once. Because the study was revised after exploratory results, we explicitly distinguish what is confirmatory from what is exploratory. Conclusions are observational associations \cite{shmueli2010explain}: we claim neither causality nor skill quality.

## 1.3 Objectives and research questions

The general objective is to investigate, reproducibly, which observable characteristics are associated with the reuse of agent skills across owners and how much this association is affected by dataset construction artifacts.

* **RQ1 — Construction biases:** to what extent do the representative-selection rule, self-copying and owner concentration confound the target and the attributes of GitSkills?
* **RQ2 — Predictability:** with attributes free of these artifacts, can cross-owner reuse and the number of owners be predicted better than baselines, and is the estimate confirmed on an owner-disjoint holdout?
* **RQ3 — Origin of the signal:** what is the contribution of skill, repository and catalog attributes, which variables are associated with the predictions, and how consistently?

## 1.4 Contributions

1. An **audit of the GitSkills representative**: we recover its rule (97.2% agreement) and show that location and history depend on the number of copies.
2. A **protocol** for modeling cross-owner reuse, with an owner-based target, a per-owner cap, grouped validation, pre-specified hypotheses with Holm correction and confirmation on a new holdout.
3. Evidence of a **modest predictive signal**, with a stable RMSLE gain across protocol variations, decomposed into blocks (skill, repository, catalog, path) and confirmed on unseen data: AUC 0.73 for cross-owner reuse.
4. A **reproducible package** that records the data revision, seeds, partitions and a lock that prevents reusing the holdout.

## 1.5 Organization

Section 2 presents related work; Section 3, the methodology; Section 4, the results; Section 5, the discussion and threats to validity; Section 6, the conclusions. The appendices cover reproduction and the use of AI tools.
