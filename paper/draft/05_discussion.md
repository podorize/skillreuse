# 5. Discussão

## 5.1 Desempenho e características dos dados

Os resultados mostram sinal preditivo, mas com alcance limitado. A representação híbrida teve o menor RMSLE médio do Ridge na validação interna; a diferença de 0,004 para `skill-only` é pequena diante do desvio-padrão entre dobras (0,028 e 0,031). O experimento sugere informação nos atributos da skill, mas não demonstra superioridade robusta do híbrido nem desempenho externo de um modelo textual. A maior correlação de Spearman de `repository-only` reforça que as métricas capturam aspectos diferentes.

A assimetria e a sobredispersão do alvo ajudam a explicar o comportamento. A maioria das skills reutilizadas aparece em poucos repositórios, enquanto uma parcela pequena concentra contagens elevadas. A referência constante teve MAE externo menor que o Random Forest (3,067 contra 3,266), embora RMSLE pior (0,714 contra 0,583). Portanto, o ganho do modelo depende da métrica escolhida. Acima do P95 do treino, o Random Forest subestimou todas as 324 observações externas, com MAE de 35,15.

## 5.2 Comparação dos modelos e vieses indutivos

O benchmark completo foi realizado na visão `repository-only`. O Random Forest foi escolhido como candidato principal pelo menor RMSLE da validação agrupada (0,5753). No teste externo, obteve 0,5833; o Gradient Boosting obteve 0,5907, mas MAE menor (3,1820 contra 3,2663). Esses valores externos avaliam candidatos já congelados e não são uma nova rodada de seleção. A variação entre três dobras e a única divisão externa limitam a precisão de comparações pequenas.

O Ridge impõe uma relação aditiva e regularizada, sendo estável e relativamente fácil de interpretar, mas tende a encolher valores extremos. KNN depende da geometria local dos atributos e teve o pior RMSLE do benchmark completo. A árvore de decisão isolada não superou os ensembles e apresentou maior variabilidade. Random Forest e Gradient Boosting capturam não linearidades e interações, mas podem explorar sinais contextuais que não generalizam para outros ecossistemas.

## 5.3 Explicabilidade e interpretação

As análises textuais exploratórias identificaram termos editoriais e sinais de templates; a robustez ao mascaramento não mostra que sejam propriedades causais da skill. No Random Forest `repository-only`, a importância por impureza destacou `repo_stars`, `history_fetched`, ausência de `commit_count` e `repo_forks`. Já a permutação agrupada colocou `repo_language` em primeiro (aumento médio de RMSLE 0,029), seguido por `commit_count` (0,014), `history_fetched` (0,013) e `repo_stars` (0,012). A divergência decorre em parte das propriedades das medidas de importância e da correlação entre atributos; o sinal de estrelas variou substancialmente entre dobras.

No cenário que varia estrelas mantendo as outras entradas, a previsão média foi não monotônica. Isso enfraquece uma regra simples sobre popularidade e reutilização, embora o cenário não seja uma estimativa causal e possa criar combinações raras de atributos. O uso de `repo_language` e histórico também pode codificar origem, cobertura da coleta ou práticas de distribuição. A interpretação do Random Forest é apenas contextual; não identifica variáveis textuais ou estruturais da skill como determinantes da reutilização.

## 5.4 Implicações para a engenharia de software

O pipeline pode apoiar análises exploratórias de autoria e descoberta, indicando convenções que merecem inspeção ou priorizando skills para auditoria manual. Ele ainda não justifica um linter automático, um ranking de qualidade ou uma intervenção de autoria. Essas aplicações exigiriam validação prospectiva, avaliação de utilidade e controle de possíveis efeitos adversos, como reforçar padrões de popularidade ou favorecer determinados ecossistemas.

Para o trabalho, a contribuição principal é metodológica: comparar modelos sob a mesma divisão agrupada, relacionar desempenho a vieses indutivos e demonstrar que métricas agregadas podem ocultar erros graves na cauda. Os resultados também delimitam o que pode ser inferido de uma medida observacional de replicação exata.

## 5.5 Ameaças à validade e limitações

`repository_count` mede a presença de conteúdo idêntico em repositórios distintos. Não mede intenção de adoção, uso efetivo, qualidade, utilidade, sucesso de execução ou transferência de conhecimento. Cópias modificadas e equivalentes semânticos não são capturados.

O agrupamento por `representative_repo_id` elimina a sobreposição desse identificador entre treino e validação, mas não garante que todos os repositórios relacionados, organizações ou cadeias de cópia estejam separados. Metadados de repositório também podem funcionar como proxies de popularidade ou origem.

O RMSLE reduz a influência de valores extremos, mas pode ocultar erros absolutos elevados. Por isso, a análise também utiliza MAE, Spearman, uma referência constante e auditoria por quantis. A subestimação acima do P95 mostra que nenhuma métrica isolada caracteriza adequadamente o desempenho. A comparação textual foi avaliada apenas com Ridge dentro do desenvolvimento; não autoriza afirmar que o modelo final explica características intrínsecas da skill.

Os dados provêm de um snapshot específico do GitHub e de um ecossistema particular de Agent Skills. A generalização para outros períodos, hosts, comunidades ou mecanismos de distribuição não é garantida. A avaliação externa reportada é válida para a divisão adotada neste estudo e não substitui replicações em outros snapshots ou ecossistemas.
