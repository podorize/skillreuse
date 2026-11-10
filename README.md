# SkillReuse

Previsibilidade do **reúso de Agent Skills entre donos** (conteúdos idênticos de `SKILL.md` sob contas distintas) e vieses de construção do corpus [GitSkills](https://huggingface.co/datasets/mvaccargiu/gitskills), revisão `ebab17454a7c236f8f26b183567f1a126f42e3f8`. Trabalho da disciplina CMP263 (UFRGS) e base do estudo para o MSR 2027 Mining Challenge.

O texto do relatório está em [`paper/draft/`](paper/draft/README.md); o Apêndice A ([`paper/draft/07_appendix_reproducibility.md`](paper/draft/07_appendix_reproducibility.md)) associa cada tabela e figura ao comando que a gera.

## Reprodução rápida (sem baixar dados brutos, ~5 min)

As tabelas derivadas necessárias (9,5 MB) estão em `data/derived/`.

```bash
uv sync
make quick        # protocolo principal (1 repetição) e re-execução dos modelos congelados no holdout
```

O holdout já foi avaliado uma vez; `make quick` o reexecuta em `results/owner/holdout_repro/` (a trava do holdout original não é afetada). Notebook para o Colab: [`notebooks/reproduce.ipynb`](notebooks/reproduce.ipynb).

## Reprodução completa

```bash
uv sync
uv run skillreuse check      # confere o ambiente
make data                    # download, amostra, divisão e alvo (requer ~7 GB em data/raw)
make owner primary           # alvo por donos, auditoria e protocolo principal
make robustness importance   # sensibilidades e interpretação
make final holdout           # tuning aninhado e holdout disjunto por dono
make holdout-eval            # avaliação ÚNICA no holdout (já executada; trava em results/owner/holdout)
make figures test
```

`uv run skillreuse --help` lista os subcomandos. Python 3.12; sementes `20261009` (amostragem) e `2027` (experimentos); validação 3×3 dobras `StratifiedGroupKFold` agrupadas por dono.

## Estrutura

| Pasta | Conteúdo |
|---|---|
| `src/skillreuse/` | `data`, `features`, `models`, `owner`, `owner_models`, `final_model`, `holdout`, `controls`, `figures`, `cli` (e módulos da primeira versão) |
| `tests/` | testes do alvo, pré-processamento, divisão e vazamento |
| `data/` | `raw/`, `interim/`, `processed/` (fora do git; reconstruíveis) |
| `results/owner/` | EDA, auditoria, protocolo principal, robustez, tuning e holdout |
| `paper/` | rascunho Markdown, projeto LaTeX/Overleaf, protocolo, status e registro de uso de IA |
| `notebooks/` | `reproduce.ipynb`, execução no Colab |

O holdout já foi avaliado uma vez (`make holdout-eval` recusa repetir sem `--allow-reevaluation`). A partição de teste original foi consumida na primeira versão do estudo e não sustenta afirmações confirmatórias.
