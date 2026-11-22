# SkillReuse

O texto de uma **agent skill** (o arquivo `SKILL.md`) prediz se outros desenvolvedores vão copiá-la? Estudo preditivo sobre o corpus [GitSkills](https://huggingface.co/datasets/mvaccargiu/gitskills) (revisão `ebab17454a7c236f8f26b183567f1a126f42e3f8`). Trabalho prático da disciplina CMP263 — Aprendizagem de Máquina (UFRGS), de Pedro Odorize e Rafael Allgayer.

- **Artigo:** PDF em [`paper/latex/main.pdf`](paper/latex/main.pdf); fonte em Markdown (inglês) em [`paper/draft/`](paper/draft/README.md).
- **Protocolo:** [`paper/protocol.md`](paper/protocol.md) (em inglês), escrito antes de treinar qualquer modelo; desvios na seção 13.
- **Uso de IA:** [`paper/supplement/ai_usage_supplement.md`](paper/supplement/ai_usage_supplement.md).

## Resultado em uma linha

Datamos pela API do GitHub as cópias de 150.000 skills e marcamos como copiada a skill que outro desenvolvedor copiou em até 60 dias após a primeira aparição, sem contar agregadores. Em 20.564 skills de autores nunca vistos, uma regressão logística sobre 22 atributos do texto alcançou AUC de 0,645 [0,624; 0,666], acima do tamanho do texto (0,595) e somando 0,049 a um modelo do repositório de origem que inclui a popularidade. O estilo de escrita é a família de atributos mais informativa (RQ2).

## Como reproduzir

**Comece pelo Colab:** [![Open In Colab](https://colab.research.google.com/assets/colab/badge.svg)](https://colab.research.google.com/github/podorize/skillreuse/blob/main/notebooks/reproduce.ipynb). Abra o notebook e execute todas as células; não é preciso instalar nada, e leva alguns minutos. Ele segue o artigo seção por seção, recalcula os números a partir dos dados publicados, aplica os modelos treinados ao conjunto de teste e termina com um placar que compara cada número com o artigo.

Para reproduzir no seu computador (Python 3.12 e [uv](https://docs.astral.sh/uv/)), clone o repositório, rode `uv sync` e escolha um dos quatro níveis. Cada nível é um único comando; os tempos foram medidos num notebook com 8 GB de memória.

| Nível | Comando | O que refaz | Precisa de | Tempo | Resultado |
|---|---|---|---|---|---|
| **1. Conferir os resultados** | `make verify` | Aplica os modelos treinados ao conjunto de teste e confere cada número do artigo | só o repositório | menos de 1 min | idêntico ao artigo |
| **2. Refazer a análise** | `make rerun` | EDA, seleção de modelos, treino, avaliação no teste, robustez e interpretação, a partir das tabelas publicadas | só o repositório | ~20 min | idêntico ao artigo |
| **3. Refazer o tratamento** | `make rebuild` | Também a amostra e a tabela do estudo (T0, autor, alvo, atributos e divisão), a partir do GitSkills, com as datas publicadas | GitSkills (~7 GB, baixado pelo comando) | ~30 min + download | idêntico ao artigo |
| **4. Refazer do zero** | `make from-scratch` | Também a coleta das datas das cópias pela API do GitHub | GitSkills e `gh` autenticado | ~8 h (quase tudo esperando a API) | parecido, não idêntico |

Os níveis 2 a 4 sobrescrevem `results/` e `models/`; depois, `make verify` confere os números e `git diff` mostra o que mudou. O nível 4 também substitui as datas publicadas: a nova coleta não sai igual, porque repositórios foram apagados ou alterados desde julho de 2026, e por isso as datas publicadas são o registro fixo do estudo. `make pdf` gera o artigo (pandoc e tectonic).

`uv run skillreuse --help` lista os subcomandos. Python 3.12. Sementes: amostra `20261121`, divisão por autor `20261122`, dobras 2027-2029, modelos, buscas e bootstrap 2027, interpretação `20261123`.

## Estrutura

| Pasta | Conteúdo |
|---|---|
| `config/` | `config.yaml`: todas as configurações do estudo (sementes, tamanhos, datas, caminhos, expressões regulares, atributos, algoritmos e espaços de busca), lidas por cada módulo de `src/skillreuse` |
| `src/skillreuse/` | `download` (download verificado), `collect` (fase 1: amostra e datas das cópias), `cohort` (fase 2: T0, autor, alvo, atributos e divisão), `features` (atributos do `SKILL.md`), `study` (fases 3-7), `stats` (Holm, Nadeau–Bengio, bootstrap por autor), `report` (figuras), `config` (lê o `config.yaml`), `cli` |
| `data/derived/` | `collection/` (amostra, datas das cópias e estrelas), `study_table.parquet` (tabela do estudo), `origin_popularity.parquet` (controle) e `aggregator_variants.parquet` (sensibilidade) |
| `models/` | modelos congelados (`joblib`), com hashes em `results/phase5/freeze.json` e `freeze_control.json` |
| `results/` | uma pasta por fase do protocolo (`phase0` a `phase7`), `figures/` e `logs/` (execução final) |
| `paper/` | `protocol.md`, `draft/` (fonte do artigo), `latex/` (projeto e PDF), `supplement/` (uso de IA), `build_latex.py` |
| `notebooks/` | guia do artigo para o Colab |
| `tests/` | invariantes do estudo, estatísticas e conferência dos números do artigo |

## Licença

Código sob a licença MIT (`LICENSE`). Os dados derivados (`data/derived/`) vêm do GitSkills (Destefanis et al., MSR 2027), distribuído sob CC-BY-4.0; não incluem o texto das skills, que permanece sob a licença de cada repositório de origem.
