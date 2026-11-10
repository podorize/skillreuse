# SkillReuse

**Caracterizacao e Classificacao da Reutilizacao de Agent Skills no GitHub**

Projeto da disciplina CMP263 - Aprendizagem de Maquina.

O [rascunho revisado do relatório da disciplina](docs/report_draft/README.md) reúne as seções, os resultados, as figuras e o apêndice de reprodução. O texto permanece em Markdown para a revisão editorial antes da formatação no template.

O objetivo e prever a quantidade observada de repositorios que reutilizam uma agent skill e
investigar quais caracteristicas textuais, estruturais e contextuais estao associadas a essa
difusao.

## Preparacao do ambiente

O projeto usa Python 3.12 e `uv` para criar o ambiente e manter as dependencias reproduziveis.

```bash
uv sync
uv run skillreuse-check
uv run pytest
```

## 🚀 Reproducao dos resultados do artigo

Para reproduzir os modelos, tabelas do teste cego de 40.000 instancias e figuras do artigo:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/podorize/skillreuse/blob/main/notebooks/reproduce_in_colab.ipynb)

```bash
# Reusa as matrizes esparsas ja construidas e regenera resultados e diagnosticos:
uv run python -m gitskills_reuse.reproduce

# (Opcional) Recria um cache textual parametrizado, sem vocabulário herdado:
uv run python -m gitskills_reuse.reproduce --from-scratch --text-max-features 10000
```

O pipeline atual nao usa o pool historico de 211 termos. Os limites textuais oficiais para a nova
avaliacao sao 5.000, 10.000 e 20.000 candidatos, com selecao posterior de 50, 100, 200 ou 500
termos dentro de cada dobra.

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

O processamento completo de tres dobras com TF-IDF e deliberadamente executado sob demanda,
pois e mais custoso do que as etapas de preparacao e analise exploratoria.

As matrizes TF-IDF/esparsas por dobra sao guardadas em `reports/modeling/ablation/` e reutilizadas
em execucoes seguintes com a mesma fonte, divisao, configuracao de atributos e codigo de
pre-processamento. Mudancas nesses elementos geram outro cache. A classificacao de intensidade usa
`saga`, tolerancia `1e-3` e ate 5.000 iteracoes por padrao; convergencia e iteracoes efetivas
ficam registradas nos resultados. A tabela parcial e atualizada ao fim de cada dobra.

## Estrutura

- `paper_cmp263/`: artigo SBC da disciplina CMP263, bibliografia e suplemento de uso de IA. O pacote Overleaf é `sbc_paper_overleaf.zip` na raiz; `REPORT_CMP263.md` é um rascunho separado.
- `paper_msr2027/`: artigo cientifico formatado para o MSR 2027 Mining Challenge (IEEE format para Overleaf).
- `configs/`: parametros de dados, features e experimentos.
- `data/`: dados brutos, intermediarios e processados (inclui cache esparso reproduzivel).
- `docs/`: metodologia, dicionario de dados e registro do uso de IA.
- `notebooks/`: exploracao, EDA e notebook interativo de 1-click no Google Colab.
- `src/gitskills_reuse/`: codigo reproduzivel do projeto.
- `tests/`: testes automatizados.
- `reports/`: figuras, tabelas e registro dos experimentos.

## Definicao do alvo

Cada instancia corresponde a um `content_hash`. O alvo principal e `repository_count` entre
skills com `reused = 1`. A metrica principal da regressao e RMSLE; MAE, erro em `log1p` e
Spearman sao metricas secundarias. `repository_count` e `occurrence_count` nunca entram como
atributos preditivos. Essas definicoes representam reutilizacao observada no snapshot, e nao
reutilizacao futura.
