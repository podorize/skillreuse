# 1. Introduction

Coding agents such as Claude Code, Codex and Cursor can be extended with agent skills: small folders centered on a SKILL.md file, a Markdown document containing a name, description, and natural-language instructions for performing tasks such as reviewing pull requests or following team conventions. \cite{agentskills_spec}. Without a dedicated package manager, developers typically reuse skills by copying them into their own repositories. Skills thus represent a distinct type of reusable software artifact. While previous reuse studies have focused primarily on code \cite{haefliger2008code,lopes2017dejavu}, this study examines the reuse of written instructions.

This raises a central question: does the textual content of a skill influence its likelihood of being copied? If textual characteristics predict copying, authors can identify effective instruction patterns, curators can prioritize skills for review, and recommendation tools can identify relevant skills. Otherwise, copying may depend primarily on who creates and shares skills and where they are published.

We develop machine learning models to predict whether other developers will copy a skill based on its text. We use GitSkills, a public dataset comprising all `SKILL.md` files found on GitHub in July 2026 \cite{gitskills2027}. We ask two questions:

* **RQ1:** Do the characteristics of a skill's text predict better than chance whether other developers will copy it?
* **RQ2:** Which text characteristics carry this signal, and in which direction?

Answering these questions requires defining what constitutes a copy. Not all copies represent independent reuse: developers may reuse their own skills, while some repositories automatically collect thousands of skills. A snapshot also hides exposure time: older skills have had more opportunities to be copied. We used the GitHub API to date copies of sampled skills and counted only those made by other developers, outside bulk collectors, within the same period after each skill first appeared. This ensures comparable exposure across skills. Since copies share identical text, we analyze the content available to developers at the time of copying.

We followed a standard machine learning workflow: exploratory data analysis, preprocessing, comparison of seven algorithms, model tuning and selection, evaluation on unseen developers, and model interpretation. To make sure the text is not just a sign of where a skill was published, we also compared it with a model of the repository where the skill first appeared.

Text attributes modestly but reliably predict copying. On 20,564 skills from unseen authors, a model using 22 text attributes achieved an AUC of 0.645. It outperformed a text-length baseline and added predictive information beyond repository popularity. Results were consistent across alternative specifications, including temporal evaluation on subsequently published skills. Explanatory description, stable vocabulary, and valid metadata were associated with higher copying rates.

The following sections present the related work (Section 2), research methodology (Section 3), empirical results (Section 4), and their implications (Section 5). Section 6 concludes the paper.
