# Appendix A — Reproducibility

All code, the dates collected through the GitHub API (already processed), the study table, the trained models and the result files are available at <https://github.com/podorize/skillreuse>. The data come from a fixed revision of GitSkills (`ebab17454a7c236f8f26b183567f1a126f42e3f8`), verified by SHA-256. The environment is Python 3.12 with every dependency pinned in `uv.lock`. All random choices use fixed seeds: `20261121` for the sample, `20261122` for the split, 2027–2029 for the cross-validation folds, 2027 for models, searches and bootstrap, and `20261123` for the interpretation sample.

Start with the Colab notebook (<https://colab.research.google.com/github/podorize/skillreuse/blob/main/notebooks/reproduce.ipynb>): it follows this paper section by section, shows the code of each step and runs it on the published data. Locally, after `uv sync`, there are four levels, one command each (times on a laptop with 8 GB of memory):

1. *Check the results* (`make verify`, under a minute): applies the trained models to the holdout and checks every number in this paper.
2. *Redo the analysis* (`make rerun`, about 20 minutes): repeats every step, from the exploratory analysis to the interpretation.
3. *Redo the processing* (`make rebuild`, about 30 minutes plus a 7 GB download): also rebuilds the sample and the study table from GitSkills.
4. *Redo from scratch* (`make from-scratch`, about 8 hours): also dates the copies again through the GitHub API; the result is close but not identical, as repositories have changed.