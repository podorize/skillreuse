# SkillReuse

Previsão da **intensidade de reutilização de Agent Skills** (número de repositórios que contêm o mesmo `SKILL.md`) a partir do corpus [GitSkills](https://huggingface.co/datasets/mvaccargiu/gitskills), revisão `ebab17454a7c236f8f26b183567f1a126f42e3f8`. Trabalho da disciplina CMP263 (UFRGS) e base do estudo para o MSR 2027 Mining Challenge.

O texto do relatório está em [`paper/draft/`](paper/draft/README.md); o Apêndice A ([`paper/draft/07_appendix_reproducibility.md`](paper/draft/07_appendix_reproducibility.md)) associa cada tabela e figura ao comando que a gera.

## Reprodução

```bash
uv sync
uv run skillreuse check      # confere o ambiente
make data                    # download, amostra, divisão e alvo (requer ~7 GB em data/raw)
make views benchmark evaluate report
make test
```

`uv run skillreuse --help` lista os subcomandos. Python 3.12; sementes `20261009` (amostragem) e `2027` (experimentos); três dobras `StratifiedGroupKFold` por `representative_repo_id`.

## Estrutura

| Pasta | Conteúdo |
|---|---|
| `src/skillreuse/` | `data`, `features`, `models`, `benchmark`, `explain`, `report` e `cli` |
| `tests/` | testes do alvo, pré-processamento, divisão e vazamento |
| `data/` | `raw/`, `interim/`, `processed/` (fora do git; reconstruíveis) |
| `results/` | tabelas, manifestos e figuras citados no relatório |
| `paper/` | rascunho Markdown, projeto LaTeX/Overleaf, protocolo, status e registro de uso de IA |
| `notebooks/` | `reproduce.ipynb`, execução no Colab |

A avaliação externa (`make evaluate`) já foi consumida pela entrega da disciplina; repeti-la serve apenas para verificar reprodutibilidade, nunca para novo ajuste.
