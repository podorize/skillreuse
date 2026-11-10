# Fase 3 — preparação e piloto de regressão de contagem

## Escopo executado

- População: 25.918 skills reutilizadas (`reused = 1`) na partição externa de treino.
- Alvo: `repository_count`, com `log1p(repository_count)` como escala alternativa de ajuste.
- Grupos: `representative_repo_id`; nenhuma dobra do piloto compartilhou grupo entre treino e validação.
- Visões: `skill-only`, `repository-only` e `hybrid`.
- Proteção contra vazamento: TF-IDF, imputação, codificação e seleção textual foram ajustados dentro de cada dobra.
- Teste externo: não foi carregado.

## Preparação do alvo

O artefato `count_target_manifest.json` confirma 25.918 linhas, 7.701 grupos e a exclusão de
`repository_count`, `reused`, `split_stratum` e as transformações do alvo das visões de preditores.
A distribuição continua fortemente sobredispersa: média 5,08, mediana 2, variância 112,22 e
razão variância/média 22,11.

## Spot-check diagnóstico

Foi executado um piloto de 3 dobras em uma amostra determinística de 3.000 linhas, com pool de
5.000 termos e seleção de 100 termos por `f_regression` sobre `log1p(count)` dentro de cada dobra.
Esse resultado **não é resultado final**; serve apenas para validar o pipeline e revelar problemas
antes da execução completa.

| Visão | Modelo | RMSLE médio | MAE médio | Spearman médio |
|---|---|---:|---:|---:|
| skill-only | baseline log | 0,853 | 7,80 | 0,000 |
| skill-only | Ridge log | 0,753 | 7,47 | 0,419 |
| repository-only | baseline log | 0,853 | 7,80 | 0,000 |
| repository-only | Ridge log | 0,796 | 272,23 | 0,412 |
| hybrid | baseline log | 0,853 | 7,80 | 0,000 |
| hybrid | Ridge log | 0,752 | 63,64 | 0,502 |

## Parecer crítico

1. O pipeline de contagem está operacional e a separação por grupo foi respeitada.
2. Os sinais textuais reduzem o RMSLE em relação ao baseline no spot-check; o híbrido teve o
   melhor RMSLE, mas a diferença para `skill-only` ainda é pequena.
3. O erro absoluto na escala original explode em algumas dobras para Ridge ajustado em log. Isso
   indica que a seleção por RMSLE sozinha não é suficiente para a interpretação da cauda; a próxima
   etapa deve comparar Poisson/Binomial Negativa, regularização e calibração/limitação das previsões.
4. Nenhuma conclusão sobre superioridade de representação ou desempenho final deve ser publicada
   com esta amostra reduzida. A execução completa precisa usar todas as 25.918 linhas e o grid
   5k/10k/20k × 50/100/200/500.

## Próximo passo

Executar o benchmark completo em etapas controladas, começando pelos modelos de contagem e pela
regressão logarítmica regularizada; registrar tempo, memória, RMSLE, MAE, MAE logarítmico e
Spearman por dobra antes de selecionar o modelo para explicabilidade.

## Atualização — grid diagnóstico e confirmação completa

O grid completo em 3.000 linhas foi usado somente para triagem. O melhor cenário foi
`skill-only`, pool de 10.000 e 500 termos (RMSLE 0,747). Essa triagem escolheu a configuração
para a confirmação sem consultar o teste externo.

Na confirmação com todas as 25.918 linhas reutilizadas e três dobras agrupadas, obtivemos:

| Visão | Modelo | RMSLE médio | MAE médio | Spearman médio |
|---|---|---:|---:|---:|
| hybrid | Ridge em log1p | **0,568** | 3,32 | 0,245 |
| skill-only | Ridge em log1p | 0,572 | 3,33 | 0,213 |
| repository-only | Ridge em log1p | 0,588 | 3,34 | 0,259 |

O híbrido é o melhor pela métrica primária, mas a vantagem sobre `skill-only` é pequena e deve
ser tratada como evidência de associação preditiva, não como efeito causal. O contexto do
repositório, isoladamente, também carrega sinal relevante.

Um Poisson completo foi executado na visão `repository-only` como controle de viabilidade:
RMSLE 0,673, pior que Ridge logarítmico (0,588). O ajuste Poisson com 500 termos textuais foi
interrompido por custo computacional do L-BFGS; portanto, não será usado como comparação principal
até que uma implementação escalável (por exemplo, solver incremental ou redução de dimensão)
seja definida.

Na etapa seguinte, o `SGDRegressor` em `log1p` divergiu nas três visões, produzindo overflow e
RMSLE infinito. O Tweedie com potência 1,5 foi executado em `repository-only` e obteve RMSLE
0,661, também inferior ao Ridge. Esses resultados reforçam o Ridge como candidato atual, mas não
substituem uma futura comparação com Binomial Negativa em representação reduzida.

Artefatos da confirmação:

- `selected_full_ridge/count_regression_pilot_summary.csv`;
- `selected_full_ridge/count_regression_pilot_folds.csv`;
- `poisson_repository_full/count_regression_pilot_summary.csv`;
- `diagnostic_grid_3000/count_regression_pilot_summary.csv`.

Conclusão operacional da Fase 3: a formulação de contagem e o protocolo agrupado estão
funcionando; Ridge em `log1p` é o candidato atual para explicabilidade, enquanto a comparação
de modelos adicionais deve ocorrer na próxima fase, sem reabrir o teste externo.

Nota de validade: como a configuração 10k/500 foi escolhida no grid diagnóstico, as métricas da
confirmação completa são estimativas internas de desenvolvimento. O desempenho generalizável
somente será reportado depois que o modelo e a configuração forem congelados e avaliados uma única
vez no teste externo.
