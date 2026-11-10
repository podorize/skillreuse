# Plano de revisão do relatório da disciplina

**Estado em 09/11/2026:** revisão analítica e textual implementada nos arquivos Markdown. A formatação no template da disciplina e a conferência visual do PDF permanecem como etapa editorial posterior, conforme o fluxo Markdown → Overleaf definido para este relatório. A tabela de importância por permutação tem apenas uma repetição por atributo e dobra; suas incertezas são indicativas.

## Escopo e decisão sobre algoritmos

O alvo permanece `repository_count` entre skills presentes em pelo menos dois repositórios. A comparação principal contém somente seis regressores treinados em `log1p(repository_count)`: Ridge, KNN, Decision Tree, Random Forest, Gradient Boosting e HistGradientBoosting. Poisson, Tweedie, SGD e Naive Bayes não integram a pesquisa apresentada no relatório. Os registros de execuções antigas permanecem apenas como histórico do repositório.

## 1. Alinhar objetivo, representações e evidência

**Prioridade: crítica.** A pergunta de pesquisa inclui características da skill, mas o benchmark completo dos seis regressores usa somente atributos de repositório.

- [x] Manter uma comparação explícita entre `skill-only`, `repository-only` e `hybrid` com o mesmo algoritmo Ridge, mesmas dobras agrupadas e mesma métrica; registrar variação entre dobras.
- [x] Apresentar o benchmark dos seis algoritmos `repository-only` como comparação de vieses indutivos para uma representação controlada.
- [x] Explicar que a análise de atributos da skill se apoia no experimento Ridge das três visões, enquanto a interpretação do Random Forest final se refere apenas ao contexto do repositório.
- [x] Revisar RQ1–RQ3 e as contribuições para que cada afirmação tenha um resultado correspondente. Se a análise textual/estrutural não sustentar uma resposta, explicitar a resposta parcial.

**Critério de conclusão:** nenhuma conclusão sobre variáveis da skill será sustentada apenas pelo Random Forest `repository-only`; tabelas e texto identificarão sempre modelo, representação, conjunto e métrica.

## 2. Corrigir a seleção e o uso do teste externo

**Prioridade: crítica.** A redação atual pode dar a entender que o teste escolheu o modelo.

- [x] Registrar a regra de escolha por RMSLE médio na validação agrupada, usando apenas o conjunto de desenvolvimento.
- [x] Informar que Random Forest e Gradient Boosting foram os candidatos levados à avaliação externa; reportar os dois resultados sem escolher retroativamente pelo desempenho do teste.
- [x] Deixar claro que os resultados do Ridge textual são de validação interna e não equivalem a uma avaliação externa desse modelo.
- [x] Não usar a partição externa já examinada para novo ajuste de hiperparâmetros ou seleção de modelo. Novas análises de desenvolvimento usarão apenas as dobras do treino.

**Critério de conclusão:** o fluxograma metodológico e as seções 3–6 distinguem seleção na validação de estimativa no teste.

## 3. Avaliar os erros e explicar o modelo escolhido

**Prioridade: crítica.** A tabela por quantis existente é do Ridge híbrido; falta a auditoria correspondente ao Random Forest final.

- [x] Calcular e apresentar, a partir das previsões externas já salvas do Random Forest, RMSLE, MAE, erro mediano e taxa de subestimação por faixas definidas no treino. Não ajustar o modelo nessa análise.
- [x] Comparar o desempenho global com uma referência simples, como a mediana do alvo do treino, usando as mesmas partições e métricas.
- [x] Produzir pelo menos um gráfico de valor observado versus predito e um gráfico de erro por faixa de intensidade para o Random Forest.
- [x] Complementar a importância por impureza com análise de permutação em validação agrupada e gráfico de cenário para estrelas; discutir correlação entre atributos e limites de interpretação.
- [x] Manter a auditoria da cauda do Ridge como resultado separado, com legenda que identifique o modelo.

**Critério de conclusão:** a discussão da cauda e das variáveis principais é apoiada por resultados do modelo ao qual se refere; a interpretação descreve direção e contexto das previsões, além da ordenação de importância.

## 4. Completar EDA e apresentação dos resultados

**Prioridade: alta.** O enunciado pede gráficos e discussão de distribuição, valores ausentes, outliers e pré-processamento.

- [x] Inserir figura da distribuição de `repository_count` e tabela de quantis, identificando treino e teste.
- [x] Quantificar ausência dos atributos de repositório, especialmente histórico de commits, e justificar indicadores de ausência e imputação.
- [x] Descrever exclusões por vazamento e a variável invariante que permaneceu no pipeline congelado; documentar os atributos de cada visão.
- [x] Adicionar média ± desvio-padrão para os seis regressores e o spot-check dos dois candidatos.
- [x] Revisar o resumo para conter objetivo, população condicional, desenho, resultado externo principal e limitação da cauda, sem prometer análises ainda não feitas.

**Critério de conclusão:** cada decisão importante de pré-processamento decorre de uma observação da EDA e cada conclusão quantitativa aponta para tabela ou figura.

## 5. Fechar reprodutibilidade e conformidade do artigo

**Prioridade: alta.** O enunciado exige link de acesso a dados públicos, descrição detalhada e código executável.

- [x] Inserir link direto da fonte dos dados e revisão/snapshot usada; descrever a seleção determinística dos 200 mil conteúdos e a condição que produz 25.918 linhas de treino e 6.479 de teste.
- [x] Completar o Apêndice A com comandos desde obtenção/preparação dos dados até geração de figuras e tabelas, dependências e sementes.
- [x] Verificar as 22 chaves `\cite{}` usadas no rascunho contra `paper_cmp263/references.bib`; nenhuma falta. Gerar a lista formatada na migração ao template.
- [x] Registrar no Apêndice B os cinco prompts preservados e a incompletude do histórico, com declaração de uso de IA para o artigo.
- [ ] Integrar figuras e tabelas ao template da disciplina, conferir coluna única, limite de 20 páginas incluindo referências e consistência entre PDF e arquivos Markdown.

**Critério de conclusão:** um leitor consegue localizar os dados, executar o pipeline documentado e associar cada número do artigo a um artefato reproduzível.

## Ordem de execução

1. Resolver as tarefas 1–2 e congelar as afirmações permitidas pelos experimentos.
2. Executar a auditoria e a interpretação do Random Forest descritas na tarefa 3, sem novo ajuste pelo teste.
3. Inserir EDA, gráficos e referência simples da tarefa 4.
4. Revisar o texto completo e concluir a tarefa 5 antes da formatação final.
