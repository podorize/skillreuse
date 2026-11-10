# Fase 1 - EDA e decisao preliminar do alvo

## Protocolo

A decisao sobre a formulacao do alvo foi baseada somente no split de desenvolvimento (`split=train`). O teste externo foi resumido para auditoria de balanceamento, mas nao participou da recomendacao. `repository_count` e a origem do alvo e nao pode ser usado como atributo preditor.

## Achados principais

- Skills reutilizadas no desenvolvimento: **25.918**.
- Mediana de `repository_count`: **2**.
- Media: **5.075**; variancia: **112.220**.
- Razao variancia/media: **22.112**, indicando forte sobredispersao.
- P95: **16.0**; P99: **55.0**; maximo: **390**.
- A classe `10+` contem **2,200** skills, mas varia de 10 a 390 reutilizacoes.

## Distribuicao ordinal

A discretizacao atual produz classes ordenadas, mas a ultima classe agrega uma cauda muito ampla. Isso melhora a estabilidade da classificacao, mas perde informacao sobre a diferenca entre reutilizacao alta e muito alta.

| split   |   intensity_code | label            |     n |   percentage |   count_min |   count_max |   count_mean |   count_median |   count_std |
|:--------|-----------------:|:-----------------|------:|-------------:|------------:|------------:|-------------:|---------------:|------------:|
| train   |                0 | 2 repositorios   | 14593 |     56.3045  |           2 |           2 |      2       |              2 |    0        |
| train   |                1 | 3-4 repositorios |  6087 |     23.4856  |           3 |           4 |      3.29916 |              3 |    0.457929 |
| train   |                2 | 5-9 repositorios |  3038 |     11.7216  |           5 |           9 |      6.28802 |              6 |    1.33923  |
| train   |                3 | 10+ repositorios |  2200 |      8.48831 |          10 |         390 |     28.7123  |             19 |   26.2117   |

## Grupos e divisao

A divisao por `representative_repo_id` e necessaria: os grupos possuem cauda longa e nao devem ser fragmentados entre treino e validacao. A tabela resume o tamanho dos grupos.

| split   | population   |   n_rows |   n_groups |   mean_rows_per_group |   median_rows_per_group |   p90_rows_per_group |   p99_rows_per_group |   max_rows_per_group |
|:--------|:-------------|---------:|-----------:|----------------------:|------------------------:|---------------------:|---------------------:|---------------------:|
| train   | all          |   160000 |      57808 |               2.76778 |                       1 |                    4 |                   17 |                 3221 |
| train   | reused       |    25918 |       7701 |               3.36554 |                       1 |                    4 |                   22 |                 3033 |

## Integridade da contagem e da divisao

`occurrence_count` e sempre maior ou igual a `repository_count`; isso e consistente com multiplas ocorrencias do mesmo conteudo dentro de um repositorio. Nao foram observados valores ausentes ou casos em que a contagem de repositorios excedesse a contagem de ocorrencias.

| population   |      n |   missing_occurrence_count |   occurrence_lt_repository |   occurrence_eq_repository |   occurrence_gt_repository |   occurrence_median |   occurrence_p99 |   occurrence_max |
|:-------------|-------:|---------------------------:|---------------------------:|---------------------------:|---------------------------:|--------------------:|-----------------:|-----------------:|
| all          | 200000 |                          0 |                          0 |                     179538 |                      20462 |                   1 |               17 |              617 |
| reused       |  32397 |                          0 |                          0 |                      20662 |                      11735 |                   3 |               85 |              617 |

A divisao externa nao possui sobreposicao de `representative_repo_id` entre treino e teste:

|   train_groups |   test_groups |   overlapping_groups |
|---------------:|--------------:|---------------------:|
|          57808 |         14568 |                    0 |

## Recomendacao para a proxima fase

Decisao da Fase 1: a pesquisa adotara **regressao de contagem** como alvo principal, preferencialmente com modelo Binomial Negativo ou regressao sobre `log1p(repository_count)`. A decisao e sustentada pela forte assimetria e sobredispersao observadas na contagem. A classificacao ordinal nao integra o protocolo principal.

O proximo passo e implementar e validar o pipeline de regressao de contagem com divisao agrupada, sem leakage, antes do tuning completo e da avaliacao no teste externo.

## Artefatos

- `reports/tables/target_eda_count_distribution.csv`
- `reports/tables/target_eda_ordinal_distribution.csv`
- `reports/tables/target_eda_group_distribution.csv`
- `reports/tables/target_eda_split_distribution.csv`
- `reports/tables/target_eda_coverage_summary.csv`
- `reports/tables/target_eda_group_overlap.csv`
- `reports/figures/target_eda_count_distribution.png`
- `reports/figures/target_eda_ordinal_distribution.png`
