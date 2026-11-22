# Supplementary Material: Use of an AI Assistant

This supplement accompanies the CMP263 paper *"Does the text of an agent skill predict its replication on GitHub? A predictive study on GitSkills"* (Appendix B). We used an AI assistant, Claude Code (Anthropic, 2026), to answer questions and suggest approaches. Both the authors and the AI assistant wrote and revised the code and the text. All AI-generated content was reviewed by the authors, who are responsible for it. The prompts below are a selection; they were written in Portuguese and are summarized here in English, with related prompts grouped together. The authors defined the research question and the objective, approved every methodological decision and are responsible for the whole work.

## Answering questions

1. "Explain the implications of splitting the test set by author."
2. "Should the repository attributes answer a research question of their own or serve as a control analysis?"
3. "Does the methodology meet the requirements of the assignment?"

## Suggesting approaches

4. "Suggest how to structure the research protocol in phases, aligned with the assignment."
5. "Suggest how to evaluate the models: the validation strategy, nested cross-validation for hyperparameter tuning, the metrics and the statistical tests for comparing models."

## Writing and revising code

6. "Help write and revise the code for the phases of the protocol: exploratory analysis, preprocessing, spot-checking, nested tuning, final models, holdout evaluation and interpretation."
7. "Help write and revise the Colab notebook and the automated tests that reproduce the results reported in the paper."

## Writing and revising text

8. "Help draft and revise the text of the paper for clarity and cohesion."

## Limitations identified

- **Wrong numbers in the text.** We compared every number of the paper with the result files, using automated tests and the Colab notebook. This found mistakes in sentences written by us or with the assistant: a percentage over the wrong denominator (11.7% of the dated copies, when it is 13.0%; we removed that paragraph), a rounding error (5.8% instead of 5.7%), a count that mixed text and repository columns (22 instead of 21), and a claim that the SHAP and permutation rankings share their top five attributes (they share four). No number is accepted without being recomputed.
- **Explanations that were not tested.** The assistant first said that k-nearest neighbors did poorly in the spot-check because of correlated and skewed attributes, but the preprocessing already handled both. After we questioned it, a diagnostic run showed that the default of five neighbors is the cause. We now state only explanations that we tested or derived.

## Reference

Anthropic. (2026). *Claude Code* (Claude Opus 5.5 and Claude Sonnet 5.5) [Large language model]. https://claude.com/claude-code
