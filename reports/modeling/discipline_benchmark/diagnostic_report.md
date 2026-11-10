# Benchmark diagnóstico para o trabalho da disciplina

## Configuração

- 3 dobras `StratifiedGroupKFold` por `representative_repo_id`;
- 3.000 linhas da população reutilizada de treino;
- representação híbrida;
- pool textual de 5.000 candidatos e 100 termos;
- regressão ajustada em `log1p(repository_count)`;
- Naive Bayes usado apenas na classificação auxiliar das faixas de intensidade.

## Resultados médios

| Modelo | RMSLE | MAE | Spearman |
|---|---:|---:|---:|
| Gradient Boosting | **0,688** | 6,47 | 0,516 |
| HistGradientBoosting | 0,702 | 6,49 | 0,470 |
| Random Forest | 0,707 | 6,73 | 0,445 |
| Ridge | 0,752 | 63,66 | 0,502 |
| KNN | 0,766 | 7,45 | 0,422 |
| Decision Tree | 0,934 | 8,58 | 0,199 |

Naive Bayes, na tarefa auxiliar de classificação das faixas, obteve macro-F1 0,201 e balanced
accuracy 0,299.

## Interpretação pedagógica

- Gradient Boosting captura relações não lineares e interações melhor que o Ridge nessa amostra.
- A árvore isolada apresenta alta variância e pior generalização.
- Random Forest reduz parte dessa variância, mas não supera o Gradient Boosting.
- KNN sofre com a dimensionalidade e a geometria da representação híbrida.
- Ridge é mais simples e explicável, mas encolhe previsões extremas.
- Naive Bayes sofre com a hipótese de independência entre termos e não é comparável diretamente
  aos regressores pela RMSLE.

Esses números são diagnósticos, não resultados finais. O benchmark completo da disciplina foi
executado na visão `repository-only` com todas as 25.918 linhas; os resultados estão em
[`full_benchmark_report.md`](../full_benchmark_report.md). A comparação textual híbrida permanece
uma análise separada do protocolo MSR, e a análise específica da cauda ainda é necessária antes
de qualquer conclusão final para o MSR.
