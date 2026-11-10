# Benchmark completo da disciplina

## Configuração

- população: 25.918 skills reutilizadas da partição externa de treino;
- unidade de análise: `content_hash`;
- agrupamento: `representative_repo_id`;
- validação: `StratifiedGroupKFold`, três dobras, seed 2027;
- alvo: `repository_count`, modelado em `log1p`;
- representação comparável: `repository-only`;
- modelos: Ridge, KNN, Decision Tree, Random Forest, Gradient Boosting e HistGradientBoosting;
- teste externo: não utilizado.

A execução híbrida completa com TF-IDF para todos os modelos não foi usada como benchmark principal da disciplina porque a vetorização repetida de milhares de documentos longos excede o orçamento computacional disponível. O resultado textual Ridge permanece reportado separadamente, pois foi confirmado no protocolo de regressão de contagem.

## Resultados médios

| Modelo | RMSLE | MAE | MAE log1p | Spearman |
|---|---:|---:|---:|---:|
| Random Forest | **0,5753 ± 0,0077** | 3,2818 ± 0,1205 | 0,4104 ± 0,0112 | 0,2873 ± 0,0384 |
| HistGradientBoosting | 0,5828 ± 0,0194 | 3,3118 ± 0,0338 | 0,4115 ± 0,0296 | 0,2551 ± 0,1288 |
| Gradient Boosting | 0,5871 ± 0,0095 | **3,1731 ± 0,1292** | **0,3871 ± 0,0159** | **0,3166 ± 0,0708** |
| Ridge | 0,5877 ± 0,0059 | 3,3409 ± 0,1561 | 0,4246 ± 0,0120 | 0,2593 ± 0,0380 |
| Decision Tree | 0,5932 ± 0,0165 | 3,3208 ± 0,0727 | 0,4030 ± 0,0215 | 0,2824 ± 0,1053 |
| KNN | 0,6103 ± 0,0271 | 3,4517 ± 0,0544 | 0,4227 ± 0,0280 | 0,2138 ± 0,0861 |

## Spot-check de hiperparâmetros

O Random Forest com profundidade máxima 12, 100 árvores, `min_samples_leaf=3` e `max_features=sqrt` apresentou o melhor RMSLE. Aumentar para 200 árvores não melhorou o resultado. No Gradient Boosting, 200 estimadores, profundidade 3, taxa de aprendizagem 0,05 e perda Huber melhoraram o RMSLE para 0,5845 e a correlação de Spearman para 0,3355.

## Interpretação pedagógica

- Random Forest foi o melhor segundo RMSLE, a métrica principal do protocolo.
- Gradient Boosting foi melhor em MAE e Spearman, indicando melhor erro absoluto e ordenação.
- HistGradientBoosting teve desempenho competitivo, mas maior variabilidade entre dobras.
- Decision Tree isolada apresentou maior instabilidade e não superou os ensembles.
- KNN foi o pior modelo, compatível com a dificuldade de usar vizinhança em atributos heterogêneos.
- Ridge permaneceu competitivo e é mais simples de interpretar, mas não venceu o benchmark não textual.

Esses resultados são válidos para a visão `repository-only` e para a partição de desenvolvimento. Eles não substituem a escolha do modelo MSR, que ainda depende da visão híbrida, da auditoria da cauda e do teste externo.

## Avaliação externa da disciplina

Após o congelamento do benchmark da disciplina, os dois candidatos selecionados foram ajustados em todas as 25.918 linhas de treino e avaliados uma única vez nas 6.479 skills reutilizadas do teste externo. Não houve sobreposição dos `representative_repo_id` entre treino e teste.

| Modelo | RMSLE | MAE | MAE log1p | Spearman |
|---|---:|---:|---:|---:|
| Random Forest | **0,5833** | 3,2663 | 0,4063 | **0,3667** |
| Gradient Boosting | 0,5907 | **3,1820** | **0,3903** | 0,2315 |

O Random Forest foi superior em RMSLE e Spearman, enquanto o Gradient Boosting manteve menor MAE. Esses resultados são a avaliação final da trilha da disciplina para a visão `repository-only`. Como o teste externo foi consultado para essa entrega, ele não deve ser reutilizado como teste cego do artigo MSR. A versão MSR deverá reservar uma nova partição externa ou uma nova coleta temporal antes da avaliação final.
