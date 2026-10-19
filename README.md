# SkillReuse

**Caracterizacao e Classificacao da Reutilizacao de Agent Skills no GitHub**

Projeto da disciplina CMP263 - Aprendizagem de Maquina.

O objetivo e classificar a reutilizacao observada de agent skills em repositorios GitHub e
investigar quais caracteristicas textuais, estruturais e contextuais estao associadas a essa
reutilizacao.

## Preparacao do ambiente

O projeto usa Python 3.12 e `uv` para criar o ambiente e manter as dependencias reproduziveis.

```bash
uv sync
uv run skillreuse-check
uv run pytest
```

## 🚀 Reproducao Rapida do Artigo (< 2 minutos)

Para reproduzir de forma ultrarrapida todos os modelos campeoes, tabelas do teste cego de 40.000 instancias ineditas e figuras em alta resolucao do artigo cientifico:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/podorize/skillreuse/blob/main/notebooks/reproduce_in_colab.ipynb)

```bash
# Executa a reproducao completa do artigo em ~1 minuto usando as matrizes pre-processadas:
uv run python -m gitskills_reuse.reproduce

# (Opcional) Para recomputar todo o pre-processamento e vetorizacao do zero:
uv run python -m gitskills_reuse.reproduce --from-scratch
```

Para iniciar o JupyterLab localmente:

```bash
uv run jupyter lab
```

## Construir a tabela analitica da amostra

A amostra oficial deve estar extraida em
`data/raw/gitskills-sample/agent_skills_sample.db`. Para validar o banco, construir o alvo e
gerar os primeiros relatorios:

```bash
uv run skillreuse-build-sample
```

O comando gera:

- `data/processed/skills_target.parquet`;
- `reports/tables/data_quality.csv`;
- `reports/tables/reuse_distribution.csv`;
- `reports/tables/sample_manifest.json`;
- `reports/figures/reuse_distribution.png`.

## Extrair atributos e congelar treino/teste

Depois da construcao da tabela analitica:

```bash
uv run skillreuse-build-features
```

O comando extrai atributos de texto e Markdown, cria uma separacao estratificada e agrupada por
repositorio e gera:

- `data/processed/skills_features.parquet`;
- `data/processed/split_assignments.parquet`;
- `reports/tables/feature_summary.csv`;
- `reports/tables/split_summary.csv`;
- `reports/tables/feature_manifest.json`;
- `reports/figures/class_distribution.png`.

## Baixar as tabelas necessarias do dataset completo

O download e fixado na revisao registrada em `configs/project.yaml`, pode ser retomado e valida
o SHA-256 de cada particao:

```bash
uv run skillreuse-download-full
```

Somente `artifacts` e `repos` sao baixadas. `artifact_siblings` nao e necessario para construir
o alvo populacional e nao e incluido nesta etapa.

## Construir a populacao e a amostra final

Depois do download completo:

```bash
uv run skillreuse-build-full
```

O comando calcula o alvo sobre a populacao completa e gera:

- `data/interim/full_population_index.parquet`, sem o texto integral;
- `data/processed/final_modeling_sample.parquet`, com 200 mil conteudos;
- `reports/tables/full_data_quality.csv`;
- `reports/tables/full_reuse_distribution.csv`;
- `reports/tables/full_population_manifest.json`;
- `reports/figures/full_reuse_distribution.png`.

A amostra e determinada por uma segunda dispersao criptografica do hash do conteudo, com
semente fixa. Ela preserva a prevalencia natural em vez de rebalancear as classes antes da
avaliacao.

## Extrair os atributos e congelar a divisao final

```bash
uv run skillreuse-build-final-features
```

Esse comando processa os textos em lotes, sem duplica-los na tabela estruturada, e gera:

- `data/processed/final_targets.parquet`, com reutilizacao binaria e intensidade condicional;
- `data/processed/final_structured_features.parquet`;
- `data/processed/final_split_assignments.parquet`;
- `reports/tables/final_feature_summary.csv`;
- `reports/tables/final_split_summary.csv`;
- `reports/tables/final_intensity_split_summary.csv`;
- `reports/tables/final_feature_quality.csv`;
- `reports/tables/final_feature_manifest.json`;
- `reports/figures/final_class_distribution.png`.

O texto integral permanece em `final_modeling_sample.parquet` para os experimentos com TF-IDF.
Os campos `occurrence_count`, `repository_count` e `reuse_intensity` nao entram na matriz de
features, pois revelariam diretamente os alvos. A intensidade e unida pelo `content_hash`
somente quando necessaria para a segunda tarefa.

## Executar a analise exploratoria

```bash
uv run skillreuse-analyze-data
```

A analise caracteriza os dados e seus atributos, calcula frequencias, medidas de tendencia
central e dispersao, assimetria, curtose, outliers, correlacoes e comparacoes por classe. As
comparacoes com o alvo usam somente o treino congelado. O relatorio principal e salvo em
`reports/data_analysis.md`, acompanhado por tabelas CSV e seis figuras reproduziveis.

Depois da revisao da divisao externa para os dois alvos, execute novamente esse comando antes
de usar comparacoes por classe no artigo. As estatisticas populacionais nao dependem do split.

## Treinar e comparar os modelos

```bash
uv run skillreuse-train
```

O experimento usa somente o treino da divisao externa e validacao cruzada agrupada por
`representative_repo_id`. Em cada dobra, imputacao, codificacao categorica e vocabulario TF-IDF
sao ajustados apenas no respectivo treino. Compara regressao logistica sem pesos e com pesos
`balanced` para (1) reutilizacao binaria e (2) intensidade entre os reutilizados. Nenhuma
reponderacao, sobreamostragem ou remocao de exemplos e aplicada ao teste. Os resultados por dobra,
resumos e protocolo ficam em `reports/modeling/`; o teste externo permanece intocado.

O processamento completo de cinco dobras com TF-IDF e deliberadamente executado sob demanda,
pois e mais custoso do que as etapas de preparacao e analise exploratoria.

As matrizes TF-IDF/esparsas por dobra sao guardadas em `reports/modeling/cache/` e reutilizadas
em execucoes seguintes com a mesma fonte, divisao e configuracao de atributos. Mudancas nas
entradas ou nos parametros de vetorizacao geram outro cache. A classificacao de intensidade usa
`saga`, tolerancia `1e-3` e ate 5.000 iteracoes por padrao; convergencia e iteracoes efetivas
ficam registradas nos resultados. A tabela parcial e atualizada ao fim de cada dobra.

## Estrutura

- `configs/`: parametros de dados, features e experimentos.
- `data/`: dados brutos, intermediarios e processados; arquivos grandes nao entram no Git.
- `docs/`: metodologia, dicionario de dados e registro do uso de IA.
- `notebooks/`: exploracao, EDA e comunicacao dos resultados.
- `src/gitskills_reuse/`: codigo reproduzivel do projeto.
- `tests/`: testes automatizados.
- `reports/`: figuras, tabelas e registro dos experimentos.

## Definicao dos alvos

Cada instancia corresponde a um `content_hash`. O alvo inicial sera:

- `0`: conteudo presente em apenas um repositorio;
- `1`: conteudo presente em dois ou mais repositorios distintos.

Para conteudos reutilizados, `reuse_intensity` assume 0 para 2 repositorios, 1 para 3-4,
2 para 5-9 e 3 para 10 ou mais. Para conteudos nao reutilizados, a intensidade e ausente.
Essas definicoes representam reutilizacao observada no snapshot, e nao reutilizacao futura.
