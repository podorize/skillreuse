# Plano MSR 2027 — estado das tarefas e próximas fases

Este plano prioriza a versão do estudo para o artigo do MSR 2027. Os artefatos mais amplos da
disciplina permanecem disponíveis, mas o protocolo do artigo usa regressão de contagem condicional
a `reused=1`, validação agrupada e explicabilidade associativa. O teste externo continua congelado
e não foi usado para seleção de atributos, modelo ou hiperparâmetros.

## Resumo executivo

- **Concluído:** protocolo, EDA do alvo, remoção do pool ativo de 211 termos, preparação do alvo
  de contagem, validação contra leakage e validação agrupada.
- **Concluído:** comparação das representações e seleção textual interna; explicabilidade global,
  local e estabilidade inicial.
- **Concluído:** auditoria de artefatos textuais e teste com texto normalizado.
- **Concluído na trilha da disciplina:** benchmark completo de seis regressores em 25.918 linhas
  com visão `repository-only`, além do spot-check dos dois melhores modelos. A classificação
  auxiliar foi retirada do escopo ativo.
- **Concluído na trilha da disciplina:** avaliação externa dos candidatos Random Forest e Gradient
  Boosting; o Random Forest obteve RMSLE 0,5833. Essa partição não pode mais ser tratada como teste
  cego do MSR.
- **Em andamento na redação:** resumo, sumário, Introdução, Trabalhos Relacionados, Metodologia
  e Resultados foram estruturados em Markdown para revisão antes da migração ao Overleaf.
- **Em andamento:** correção/avaliação da subestimação na cauda de alta reutilização.
- **Pendente:** congelamento final, teste externo e redação MSR.
- **Resultado candidato atual:** Ridge ajustado em `log1p(repository_count)`, configuração
  híbrida com pool de 10.000 candidatos e 500 termos; RMSLE interno agrupado = 0,568.

## Fase 0 — protocolo e desenho

| ID | Tarefa | Estado | Evidência/observação |
|---|---|---|---|
| F0.1 | Definir `repository_count` condicional a `reused=1` como alvo principal | **Concluída** | `docs/research_protocol.md` |
| F0.2 | Fixar `representative_repo_id` como grupo | **Concluída** | Sobreposição treino–teste = 0 |
| F0.3 | Congelar regras de leakage | **Concluída** | Alvos, identificadores e estratos excluídos |
| F0.4 | Definir grid textual 5k/10k/20k × 50/100/200/500 | **Concluída** | Seleção refeita dentro de cada dobra |
| F0.5 | Definir métricas | **Concluída** | RMSLE primária; MAE, MAE logarítmico e Spearman secundárias |
| F0.6 | Congelar família/modelo final | **Pendente** | Depende da comparação de modelos |

## Fase 1 — EDA e decisão do alvo

| ID | Tarefa | Estado | Evidência/observação |
|---|---|---|---|
| F1.1 | Auditar distribuição de `repository_count` | **Concluída** | Média 5,08; mediana 2; máximo 390 |
| F1.2 | Quantificar sobredispersão | **Concluída** | Razão variância/média = 22,11 |
| F1.3 | Auditar cobertura e consistência dos alvos | **Concluída** | `occurrence_count >= repository_count`; sem inconsistências |
| F1.4 | Decidir entre ordinal e regressão de contagem | **Concluída** | Regressão de contagem adotada |
| F1.5 | Registrar parecer da fase | **Concluída** | `reports/target_eda.md` |

## Fase 2 — reconstrução do pipeline sem os 211 termos

| ID | Tarefa | Estado | Evidência/observação |
|---|---|---|---|
| F2.1 | Retirar o cache ativo baseado em 211 termos | **Concluída** | Cache antigo movido para arquivo de auditoria |
| F2.2 | Implementar preparação do alvo de contagem | **Concluída** | `count_targets.py`; 25.918 linhas e 7.701 grupos |
| F2.3 | Implementar visões skill-only/repository-only/hybrid | **Concluída** | Manifesto registra 36, 8 e 44 preditores brutos |
| F2.4 | Validar exclusão do alvo das três visões | **Concluída** | Testes automatizados aprovados |
| F2.5 | Ajustar seleção textual para alvo contínuo | **Concluída** | `f_regression` sobre `log1p(count)` |
| F2.6 | Limpar completamente scripts históricos binários | **Pendente de revisão** | Não participam do protocolo de contagem, mas permanecem para auditoria histórica |

## Fase 3 — benchmark agrupado de regressão de contagem

| ID | Tarefa | Estado | Evidência/observação |
|---|---|---|---|
| F3.1 | Construir dobras agrupadas sem sobreposição | **Concluída** | 3 dobras; `group_overlap = 0` |
| F3.2 | Executar grid diagnóstico em 3.000 linhas | **Concluída** | `diagnostic_grid_3000/` |
| F3.3 | Selecionar configuração candidata | **Concluída** | 10k candidatos, 500 termos; skill-only foi melhor no diagnóstico |
| F3.4 | Confirmar configuração na população completa de treino | **Concluída** | `selected_full_ridge/`; 25.918 linhas |
| F3.5 | Comparar skill-only/repository-only/hybrid | **Concluída** | Hybrid RMSLE 0,568; skill-only 0,572; repository-only 0,588 |
| F3.6 | Testar Poisson | **Parcialmente concluída** | Repository-only RMSLE 0,673; textual completo inviável com L-BFGS atual |
| F3.7 | Comparar Binomial Negativa/Tweedie ou solver escalável | **Parcialmente concluída** | Tweedie repository-only RMSLE 0,661; SGD divergiu; Ridge permanece superior |
| F3.8 | Congelar configuração e modelo para explicabilidade | **Suspensa** | A auditoria da cauda mostrou subestimação sistemática em `>P95` |

## Fase 4 — explicabilidade e robustez

| ID | Tarefa | Estado | Critério de conclusão |
|---|---|---|---|
| F4.1 | Treinar modelo provisório no treino externo | **Concluída** | Ridge híbrido ajustado sem teste externo |
| F4.2 | Importância global de atributos estruturais e contextuais | **Concluída** | `global_feature_importance.csv` |
| F4.3 | Explicabilidade textual dos termos selecionados | **Concluída com ressalvas** | Termos recorrentes incluem datas/URLs e exigem auditoria |
| F4.4 | Explicações locais de casos de alta reutilização | **Concluída inicialmente** | `local_explanations.csv` |
| F4.5 | Estabilidade por dobra e bootstrap | **Parcialmente concluída** | Estabilidade em 3 dobras; bootstrap ainda pendente |
| F4.6 | Auditoria de erros e cauda longa | **Concluída inicialmente** | `>P95`: RMSLE 1,692 e subestimação em 100% dos casos |
| F4.7 | Comparar estratégias para a cauda | **Pendente** | Pesos, calibração e modelos sobredispersos; sem consultar o teste |

## Fase 5 — trilha MSR: correção da cauda e congelamento

| ID | Tarefa | Estado | Critério de conclusão |
|---|---|---|---|
| F5.1 | Definir métricas de decisão MSR | **Em andamento** | RMSLE global + guardrails P90/P95/P99 |
| F5.2 | Comparar pesos/calibração/modelo sobredisperso | **Pendente** | Apenas treino agrupado; teste externo intocado |
| F5.3 | Escolher compromisso global–cauda | **Pendente** | Modelo não será congelado se subestimar sistematicamente P95 |
| F5.4 | Congelar pipeline, seed, vocabulário e hiperparâmetros | **Pendente** | Manifesto completo e reprodutível |

## Trilha D — trabalho da disciplina

Esta trilha amplia a comparação de modelos para fins pedagógicos. Ela mantém o alvo de regressão
e as divisões agrupadas. A comparação ativa contém apenas Ridge, KNN, árvore de decisão, Random
Forest, Gradient Boosting e HistGradientBoosting.

| ID | Tarefa | Estado | Evidência/observação |
|---|---|---|---|
| D1 | Implementar benchmark agrupado de regressão | **Concluída** | Ridge, KNN, árvore, Random Forest e Gradient Boosting |
| D2 | Executar benchmark diagnóstico em 3.000 linhas | **Concluída** | Gradient Boosting teve RMSLE médio 0,688 |
| D3 | Retirar a classificação auxiliar do escopo ativo | **Concluída** | O benchmark da disciplina contém somente regressores; artefatos antigos são históricos |
| D4 | Executar comparação completa no treino de desenvolvimento | **Pendente** | Mesmas dobras e configuração documentada |
| D5 | Relacionar desempenho aos vieses dos modelos | **Concluída inicialmente** | `reports/modeling/discipline_benchmark/full_benchmark_report.md` |
| D6 | Produzir seção específica para o relatório da disciplina | **Pendente** | Comparação, gráficos, limitações e reflexão |

### Entregáveis e prazo da disciplina

O Enunciado-4 exige um artigo científico, o código/pipeline executável e uma apresentação oral.
O artigo deve cobrir problema, EDA, pré-processamento, treinamento, validação, resultados,
interpretação e conclusões; o código deve incluir instruções de execução e permitir avaliação
crítica do fluxo completo. O prazo indicado no enunciado para código e artigo é **22/11/2026,
às 23:59h**; a apresentação está prevista para **01/12/2026**.

| ID | Tarefa de fechamento | Estado | Próxima ação |
|---|---|---|---|
| D7 | Executar benchmark completo com todos os modelos | **Concluída** | 25.918 linhas; visão `repository-only`; três dobras agrupadas |
| D8 | Fazer spot-check de hiperparâmetros dos dois melhores modelos | **Concluída** | Random Forest e Gradient Boosting; `spot_check_summary.csv` |
| D9 | Selecionar um modelo para interpretação | **Concluída provisoriamente** | Random Forest por RMSLE; Gradient Boosting mantido como contraste |
| D10 | Executar explicabilidade do modelo escolhido | **Concluída inicialmente** | `final_repository_only/*_feature_importance.csv` |
| D11 | Avaliar uma vez no teste externo | **Concluída** | Random Forest e Gradient Boosting; grupos disjuntos |
| D12 | Fechar artigo, código, README e suplemento de IA | **Em andamento** | Markdown completo; migração ao Overleaf pendente |
| D13 | Preparar apresentação e participação na revisão por pares | **Pendente** | Apresentação em 01/12; cada aluno revisa dois artigos |

## Trilha R — desenvolvimento do relatório da disciplina

Esta trilha controla a redação do relatório em Markdown antes da integração ao template do
Overleaf. As seções que dependem de resultados finais permanecem provisórias até a conclusão do
benchmark completo e da avaliação externa.

| ID | Tarefa | Estado | Evidência/observação |
|---|---|---|---|
| R1 | Aprovar resumo e sumário | **Concluída** | Estrutura alinhada ao Enunciado-4 |
| R2 | Redigir Introdução | **Concluída inicialmente** | `docs/report_draft/01_introduction.md` |
| R3 | Redigir Trabalhos Relacionados | **Concluída inicialmente** | `docs/report_draft/02_related_work.md` |
| R4 | Redigir Metodologia | **Concluída inicialmente** | `docs/report_draft/03_methodology.md` |
| R5 | Redigir Resultados preliminares | **Atualizada** | Inclui benchmark completo da disciplina e spot-check; falta teste externo |
| R6 | Revisar consistência entre seções e protocolo | **Pendente** | Remover afirmações obsoletas e alinhar números |
| R7 | Redigir Discussão | **Concluída inicialmente** | `docs/report_draft/05_discussion.md`; atualizar após teste externo |
| R8 | Redigir Ameaças à validade e limitações | **Concluída inicialmente** | Incorporadas à subseção 5.5 |
| R9 | Redigir Conclusão | **Concluída inicialmente** | `docs/report_draft/06_conclusion.md`; atualizar após resultados finais |
| R10 | Migrar o conteúdo aprovado para o Overleaf | **Pendente** | Preservar referências, figuras e tabelas |
| R11 | Revisar paginação, referências e reprodutibilidade | **Pendente** | Gerar PDF final e pacote de código |

## Fase 6 — trilha MSR: avaliação final e explicabilidade

| ID | Tarefa | Estado | Critério de conclusão |
|---|---|---|---|
| F6.1 | Treinar o modelo congelado no treino externo | **Pendente** | Sem consultar o teste |
| F6.2 | Avaliar uma única vez no teste externo | **Pendente — nova partição necessária** | A partição anterior foi consumida pela entrega da disciplina |
| F6.3 | Reportar métricas globais e por quantis | **Pendente** | RMSLE, MAE, Spearman, viés e subestimação |
| F6.4 | Recalcular explicabilidade do modelo final | **Pendente** | Coeficientes, termos, contribuições locais |
| F6.5 | Estabilidade por dobra/bootstrap | **Pendente** | Recorrência e consistência de sinal |
| F6.6 | Analisar erros e limitações | **Pendente** | Cauda, artefatos, generalização e ausência de causalidade |

## Fase 7 — trilha MSR: redação e submissão

| ID | Tarefa | Estado | Critério de conclusão |
|---|---|---|---|
| F7.1 | Reescrever método e protocolo no artigo | **Pendente** | Sem referências ao pipeline ordinal/211 ativo |
| F7.2 | Inserir resultados finais e intervalos/variação | **Pendente** | Apenas resultados do protocolo congelado |
| F7.3 | Redigir contribuição e limitações | **Pendente** | Afirmações associativas, sem causalidade |
| F7.4 | Atualizar figuras, tabelas e referências | **Pendente** | Consistentes com o teste externo |
| F7.5 | Revisar reprodutibilidade e pacote de artefatos | **Pendente** | Seeds, manifests, versões e comandos |
| F7.6 | Revisão final para o challenge MSR 2027 | **Pendente** | Escopo enxuto e contribuição claramente delimitada |

## Próxima sequência recomendada

1. **Disciplina:** decidir D9 com base em RMSLE, MAE, Spearman e comportamento da cauda.
2. **Relatório:** revisar R2–R5 e consolidar R6–R9 com os resultados completos.
3. **Disciplina:** concluir D12, migrar o texto ao Overleaf e fechar o pacote até 22/11.
4. Preparar a apresentação oral e a revisão por pares.
5. **MSR:** depois da entrega da disciplina, executar F5–F7 com o modelo e protocolo refinados.

## Limitação atual

As métricas atuais são internas à partição de desenvolvimento. A configuração híbrida 10k/500 foi
selecionada e confirmada internamente, mas ainda não é o resultado generalizável final. O modelo
atual subestima 100% dos casos acima do P95; por isso, o teste externo não deve ser executado antes
da conclusão de F5.2–F5.4.
