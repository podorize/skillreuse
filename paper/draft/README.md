# Report draft (Markdown source, English)

Textual source of the CMP263 report, converted to LaTeX by `paper/build_latex.py` into `paper/latex/` (SBC template, Overleaf).

- `00_summary_and_outline.md`: abstract and contents;
- `01`–`06`: Introduction, Related Work, Materials and methods, Results, Discussion (includes limitations) and Conclusion;
- `07_appendix_reproducibility.md`: commands and origin of the numbers;
- `08_appendix_ai_usage.md`: AI use.

Figures come from `results/figures/` and `\cite{}` keys from `paper/latex/references.bib`. Protocol: `../protocol.md`. Every number is checked against the result files by `tests/test_report_numbers.py`.
