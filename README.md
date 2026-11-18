# SkillReuse

O texto de uma **agent skill** (o arquivo `SKILL.md`) prediz se outros desenvolvedores vão copiá-la? Estudo preditivo sobre o corpus [GitSkills](https://huggingface.co/datasets/mvaccargiu/gitskills) (revisão `ebab17454a7c236f8f26b183567f1a126f42e3f8`). Trabalho prático da disciplina CMP263 — Aprendizagem de Máquina (UFRGS), de Pedro Odorize e Rafael Allgayer.

- **Artigo:** PDF em [`paper/latex/main.pdf`](paper/latex/main.pdf); fonte em Markdown (inglês) em [`paper/draft/`](paper/draft/README.md).
- **Protocolo:** [`paper/protocol.md`](paper/protocol.md) (em inglês), escrito antes de treinar qualquer modelo; desvios na seção 13.
- **Uso de IA:** [`paper/supplement/ai_usage_supplement.md`](paper/supplement/ai_usage_supplement.md).

## Resultado em uma linha

Datamos pela API do GitHub as cópias de 150.000 skills e marcamos como copiada a skill que outro desenvolvedor copiou em até 60 dias após a primeira aparição, sem contar agregadores. Em 20.564 skills de autores nunca vistos, uma regressão logística sobre 22 atributos do texto alcançou AUC de 0,645 [0,624; 0,666], acima do tamanho do texto (0,595) e somando 0,049 a um modelo do repositório de origem que inclui a popularidade. O estilo de escrita é a família de atributos mais informativa (RQ2).

## Reprodução rápida (sem dados brutos nem API)

As datas das cópias já coletadas estão em `data/derived/collection/`, a tabela do estudo em `data/derived/study_table.parquet` e os modelos congelados em `models/`.

```bash
uv sync
make quick        # spot-checking (1 repetição) e conferência dos modelos congelados no holdout
make test         # invariantes do estudo e conferência dos números do artigo
```

Notebook do Colab, que acompanha o artigo seção a seção: [`notebooks/reproduce.ipynb`](notebooks/reproduce.ipynb).

## Reprodução completa (dados brutos, ~7 GB)

```bash
uv sync
make data         # download verificado do GitSkills
make collect      # fase 1: amostra e datas das cópias (API do GitHub, gh autenticado; horas)
make study        # fases 2-5: autores e divisão, EDA e janela, seleção, modelos finais e congelamento
make holdout      # fase 6: avaliação única (trava em results/phase6/) e controle
make robustness   # fase 7: sensibilidades, interpretação e figuras
make pdf          # LaTeX e PDF do artigo (pandoc e tectonic)
```

`uv run skillreuse --help` lista os subcomandos. Python 3.12. Sementes: amostra `20261121`, divisão por autor `20261122`, dobras 2027-2029, modelos, buscas e bootstrap 2027, interpretação `20261123`.

## Estrutura

| Pasta | Conteúdo |
|---|---|
| `src/skillreuse/` | `download` (download verificado), `collect` (fase 1: amostra e datas das cópias), `cohort` (fase 2: T0, autor, alvo, atributos e divisão), `features` (atributos do `SKILL.md`), `study` (fases 3-7), `stats` (Holm, Nadeau–Bengio, bootstrap por autor), `report` (figuras), `cli` |
| `data/derived/` | `collection/` (amostra, datas das cópias e estrelas), `study_table.parquet` (tabela do estudo), `origin_popularity.parquet` (controle) e `aggregator_variants.parquet` (sensibilidade) |
| `models/` | modelos congelados (`joblib`), com hashes em `results/phase5/freeze.json` e `freeze_control.json` |
| `results/` | uma pasta por fase do protocolo (`phase0` a `phase7`), `figures/` e `logs/` (execução final) |
| `paper/` | `protocol.md`, `draft/` (fonte do artigo), `latex/` (projeto e PDF), `supplement/` (uso de IA), `build_latex.py` |
| `notebooks/` | guia do artigo para o Colab |
| `tests/` | invariantes do estudo, estatísticas e conferência dos números do artigo |

## Licença

Código sob a licença MIT (`LICENSE`). Os dados derivados (`data/derived/`) vêm do GitSkills (Destefanis et al., MSR 2027), distribuído sob CC-BY-4.0; não incluem o texto das skills, que permanece sob a licença de cada repositório de origem.
