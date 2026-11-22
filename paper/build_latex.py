"""Gera o projeto LaTeX (template SBC) a partir do rascunho em Markdown de ``paper/draft``.

O Markdown e a fonte editavel; as figuras sao copiadas de ``results/figures``. Nenhum modelo
e treinado. Requer ``pandoc``; o PDF e compilado com ``make pdf`` (``tectonic``).
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DRAFT = ROOT / "paper" / "draft"
PAPER = ROOT / "paper" / "latex"
SECTIONS = (
    "01_introduction",
    "02_related_work",
    "03_methodology",
    "04_results",
    "05_discussion",
    "06_conclusion",
)
APPENDICES = ("07_appendix_reproducibility", "08_appendix_ai_usage")
FIGURE_DIR = PAPER / "figures"
# figuras pequenas nao precisam da largura toda (evita estouro de pagina)
FIGURE_WIDTH: dict[str, float] = {"eda_univariate.png": 0.62, "eda_attributes.png": 0.95,
                                   "shap.png": 0.9, "importance.png": 0.85,
                                   "eda.png": 0.95, "model_selection.png": 0.85,
                                   "results.png": 0.95, "workflow.png": 0.8}


def preprocess(markdown: str) -> str:
    """Remove manual section numbers and make figures portable in LaTeX."""
    markdown = re.sub(r"(?m)^(#{1,2})\s+(?:\d+(?:\.\d+)?\.?|[AB]\.\d+)\s+", r"\1 ", markdown)
    markdown = re.sub(r"(?m)^# Appendix [AB] —\s*", "# ", markdown)

    figure_labels = set()
    def figure(match: re.Match[str]) -> str:
        caption, relative_path = match.groups()
        source = (DRAFT / relative_path).resolve()
        if not source.is_file() or source.suffix.lower() != ".png":
            raise FileNotFoundError(source)
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", source.stem).strip("-").lower()
        label = f"fig:{slug}"
        if label in figure_labels:
            raise ValueError(f"Duplicate figure label: {label}")
        figure_labels.add(label)
        caption = re.sub(r"`([^`]+)`", lambda m: r"\texttt{" + m.group(1).replace("_", r"\_") + "}",
                         caption).replace("%", r"\%")
        output_name = f"report_{source.name}"
        width = FIGURE_WIDTH.get(source.name, 1.0)
        shutil.copy2(source, FIGURE_DIR / output_name)
        return (
            "\n\\begin{figure}[!tbp]\n\\centering\n"
            f"\\includegraphics[width={width}\\linewidth]{{figures/{output_name}}}\n"
            f"\\caption{{{caption}}}\n"
            f"\\label{{{label}}}\n"
            "\\end{figure}\n"
        )

    markdown = re.sub(r"!\[([^]]+)\]\(([^)]+\.png)\)", figure, markdown)

    # Relative Markdown hyperlinks would resolve outside the Overleaf archive.
    # Keep their readable labels as code; the replication package URL is in
    # the title footnote and the appendix describes their repository paths.
    def local_link(match: re.Match[str]) -> str:
        label, target = match.groups()
        return match.group(0) if target.startswith(("https://", "http://")) else f"`{label}`"

    markdown = re.sub(r"(?<!!)\[([^]]+)\]\(([^)]+)\)", local_link, markdown)
    return markdown


def convert(markdown: str) -> str:
    result = subprocess.run(
        ["pandoc", "-f", "markdown+raw_tex", "-t", "latex", "--no-highlight"],
        input=markdown,
        text=True,
        check=True,
        capture_output=True,
    )
    return ascii_symbols(result.stdout)


SYMBOLS = {
    "±": r"\ensuremath{\pm}",
    "−": r"\ensuremath{-}",
    "×": r"\ensuremath{\times}",
    "→": r"\ensuremath{\rightarrow}",
    "ρ": r"\ensuremath{\rho}",
    "≈": r"\ensuremath{\approx}",
    "≥": r"\ensuremath{\geq}",
    "≤": r"\ensuremath{\leq}",
    "–": "--",
}


def ascii_symbols(latex: str) -> str:
    """Replace Unicode symbols by LaTeX macros, independent of the input encoding."""
    for char, macro in SYMBOLS.items():
        latex = latex.replace(char, macro)
    return latex


def abstract() -> str:
    source = (DRAFT / "00_summary_and_outline.md").read_text(encoding="utf-8")
    body = source.split("## Abstract\n", 1)[1].split("\n## Contents", 1)[0].strip()
    return convert(body).strip()


def main() -> None:
    (PAPER / "sections").mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for name in (*SECTIONS, *APPENDICES):
        text = (DRAFT / f"{name}.md").read_text(encoding="utf-8")
        latex = convert(preprocess(text))
        (PAPER / "sections" / f"{name}.tex").write_text(latex, encoding="utf-8")

    preamble = r"""\documentclass[12pt]{article}
\usepackage{sbc-template}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[english]{babel}
\usepackage{graphicx}
\usepackage{float}
\usepackage{booktabs,longtable,array,calc}
\usepackage{amsmath,amssymb}
\usepackage{url}
\usepackage[hidelinks]{hyperref}
% hyperref puts the [Author year] label back in front of each reference; the SBC style lists
% the references without labels, with a hanging indent
\makeatletter\AtBeginDocument{\renewcommand\@biblabel[1]{}\let\@BIBLABEL\@biblabel}\makeatother
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\setlength{\emergencystretch}{3em}
\sloppy

\title{Does the text of an agent skill predict its replication on GitHub? A predictive study on GitSkills\footnote{Data and code: \protect\url{https://github.com/podorize/skillreuse}.}}
\author{Pedro de Paula Odorize\inst{1}, Rafael Borges Allgayer\inst{1}}
\address{Instituto de Informática -- Universidade Federal do Rio Grande do Sul (UFRGS)\\
  Porto Alegre, RS -- Brasil
}

\begin{document}
\maketitle
\thispagestyle{empty}
\begin{abstract}
"""
    body = [preamble, abstract(), r"\end{abstract}",
            r"\noindent\textbf{Keywords:} agent skills; software reuse; machine learning; mining software repositories; interpretability."]
    body.extend(f"\\input{{sections/{name}.tex}}" for name in SECTIONS)
    body.extend([r"\bibliographystyle{sbc}", r"\bibliography{references}", r"\appendix"])
    body.extend(f"\\input{{sections/{name}.tex}}" for name in APPENDICES)
    body.append(r"\end{document}")
    (PAPER / "main.tex").write_text("\n\n".join(body) + "\n", encoding="utf-8")
    print("Generated", PAPER / "main.tex")


if __name__ == "__main__":
    main()
