"""Gera um unico HTML legivel do relatorio (figuras embutidas, citacoes resolvidas).

Uso: ``python paper/build_html.py`` -> ``paper/report.html``. E apenas uma versao de leitura; a
versao para submissao e o LaTeX gerado por ``paper/build_latex.py``.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRAFT = ROOT / "paper" / "draft"
BIB = ROOT / "paper" / "latex" / "references.bib"
OUT = ROOT / "paper" / "report.html"
PARTS = ("01_introduction", "02_related_work", "03_methodology", "04_results",
         "05_discussion", "06_conclusion", "07_appendix_reproducibility",
         "08_appendix_ai_usage")
TITLE = ("Cross-owner reuse of agent skills on GitHub: predictability and construction biases "
         "of GitSkills")
CSS = """body{max-width:52rem;margin:2rem auto;padding:0 1rem;font:16px/1.55 Georgia,serif;
color:#222}h1,h2{font-family:system-ui,sans-serif}table{border-collapse:collapse;margin:1rem 0;
font-size:.85rem}td,th{border-bottom:1px solid #ccc;padding:.25rem .5rem}th{border-bottom:2px solid
#444}code{font-size:.85em;background:#f3f3f3;padding:0 .2em}figure{margin:1rem 0}
figcaption,caption{font-size:.85rem;color:#444}img{max-width:100%}"""


def cites(text: str) -> str:
    return re.sub(r"\\cite\{([^}]*)\}",
                  lambda m: "[" + "; ".join("@" + k.strip() for k in m.group(1).split(",")) + "]",
                  text)


def main() -> None:
    abstract = (DRAFT / "00_summary_and_outline.md").read_text(encoding="utf-8")
    abstract = abstract.split("## Abstract\n", 1)[1].split("\n## Contents", 1)[0].strip()
    body = "\n\n".join(cites((DRAFT / f"{p}.md").read_text(encoding="utf-8")) for p in PARTS)
    source = f"---\ntitle: \"{TITLE}\"\nabstract: |\n" + "\n".join(
        "  " + line for line in abstract.splitlines()) + "\n---\n\n" + body
    css = OUT.with_suffix(".css")
    css.write_text(CSS, encoding="utf-8")
    subprocess.run(
        ["pandoc", "-f", "markdown+raw_tex", "-t", "html5", "--standalone", "--embed-resources",
         "--citeproc", f"--bibliography={BIB}", "--metadata", "link-citations=true",
         "--toc", "--toc-depth=2", "--mathml", f"--css={css}", f"--resource-path={DRAFT}",
         "-o", str(OUT)], input=source, text=True, check=True, cwd=DRAFT)
    css.unlink()
    print("Generated", OUT)


if __name__ == "__main__":
    main()
