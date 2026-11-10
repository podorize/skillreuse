# 3. Metodologia

## 3.1 Desenho da pesquisa

Este estudo adota um desenho empírico quantitativo, preditivo e interpretável. O objetivo não é estimar um efeito causal da forma de escrever uma skill sobre sua reutilização, mas investigar se características observáveis contêm informação preditiva sobre a intensidade de sua replicação entre repositórios. Essa distinção segue a separação entre explicação e previsão proposta na literatura de modelagem estatística \cite{shmueli2010explain}.

O pipeline compara três representações com Ridge nas mesmas dobras agrupadas e, em experimento distinto, seis regressores sob a representação `repository-only`. Vocabulário, seleção de termos, imputação e codificação são ajustados dentro das dobras. Hiperparâmetros e candidatos ao teste foram escolhidos pelos resultados de validação do desenvolvimento. O teste externo foi consultado uma vez para estimar o desempenho dos dois candidatos congelados; suas previsões salvas são usadas depois apenas para análise descritiva.

## 3.2 Dados e unidade de análise

A fonte é o [corpus público GitSkills](https://huggingface.co/datasets/mvaccargiu/gitskills), revisão `ebab17454a7c236f8f26b183567f1a126f42e3f8` \cite{gitskills2027}. A amostra contém 200.000 conteúdos distintos selecionados deterministicamente pela ordenação `md5(content_hash || ':20261009')`. A unidade de análise é um conteúdo único de `SKILL.md`, identificado por `content_hash`.

A deduplicação ocorre antes da divisão dos dados. Assim, quando o mesmo conteúdo aparece em vários repositórios, ele contribui com uma única observação analítica. O número de repositórios distintos em que o conteúdo aparece é calculado sobre todas as ocorrências disponíveis e utilizado exclusivamente para definir o alvo. Identificadores, contagens derivadas do alvo e atributos que revelam diretamente a atribuição de treino ou teste não são utilizados como preditores.

Cada conteúdo possui um `representative_repo_id`, utilizado como grupo de dependência. Esse identificador representa o contexto de repositório disponível na tabela de atributos e não garante, por si só, a separação de todos os repositórios relacionados ou de todas as cópias quase idênticas. Essa limitação é considerada na interpretação da generalização.

## 3.3 Definição do alvo

O alvo principal é `repository_count`, definido como o número de repositórios distintos que contêm exatamente o mesmo conteúdo de `SKILL.md`. A modelagem é condicional a `reused = 1`, isto é, considera apenas conteúdos presentes em pelo menos dois repositórios. Dessa forma, a pergunta operacional é:

> Entre as skills reutilizadas, em que medida as características observadas permitem prever quantos repositórios as reutilizam?

A escala de contagem é preservada como desfecho principal. Utiliza-se `log1p(repository_count)` durante o treinamento dos regressores e realiza-se a transformação inversa para a avaliação na escala original. Os estratos derivados da contagem são usados somente para compor as partições agrupadas; não são preditores nem alvos dos modelos.

A análise exploratória mostrou forte assimetria positiva e sobredispersão: a média do alvo é aproximadamente 5,08, a mediana é 2, a variância é aproximadamente 112,22 e a razão variância/média é aproximadamente 22,11. Esses resultados motivam o treinamento na escala `log1p` e a avaliação tanto do erro global quanto dos erros na cauda da distribuição.

## 3.4 Divisão dos dados e controle de dependência

A avaliação tem duas camadas. A partição externa separa 160.000 conteúdos para desenvolvimento e 40.000 para teste, com grupos de `representative_repo_id` disjuntos. Após condicionar a `reused = 1`, restam 25.918 e 6.479 observações, respectivamente. O teste externo não participou da seleção de atributos, algoritmos ou hiperparâmetros. Os dois resultados externos já obtidos são apresentados integralmente.

Segundo, o conjunto de desenvolvimento é avaliado por validação cruzada agrupada, atualmente com três dobras. O agrupamento impede que observações pertencentes ao mesmo `representative_repo_id` apareçam simultaneamente no treino e na validação de uma dobra. A estratificação é utilizada para manter, tanto quanto possível, a distribuição do alvo e dos estratos de reutilização entre as dobras. Essa estratégia segue a recomendação de separar amostras relacionadas em problemas com estrutura hierárquica \cite{saeb2017need,roberts2017cross}.

Em cada dobra, todas as operações ajustáveis são aprendidas somente no subconjunto de treino: imputação, escalonamento, codificação categórica, vocabulário TF-IDF, seleção de termos e ajuste do modelo. Esse procedimento reduz o risco de vazamento de informação e evita que a validação influencie diretamente a construção das variáveis \cite{kaufman2012leakage}.

## 3.5 Construção das variáveis

As variáveis são organizadas em três blocos:

1. **Atributos da skill:** medidas de comprimento e diversidade lexical, estrutura Markdown, front matter, localização, nome de arquivo e presença de scripts, referências e arquivos relacionados.
2. **Atributos textuais:** unigramas e bigramas do conteúdo de `SKILL.md`, representados por TF-IDF com frequência de termo sublinear.
3. **Atributos do repositório:** linguagem, licença, estrelas, forks, commits, estado de fork e indicadores de coleta de metadados e histórico. A localização da skill pertence ao bloco estrutural.

Valores ausentes categóricos recebem uma categoria explícita. Valores numéricos são imputados pela mediana aprendida no treino da dobra, com indicadores para colunas com ausência, e depois escalonados. A ausência de histórico é interpretada como falta de coleta, não como ausência de atividade. `content_hash`, `representative_repo_id`, `repository_count`, `occurrence_count`, `reused`, `reuse_intensity`, estrato e partição são excluídos dos preditores.

Não foi aplicado corte ou winsorização às contagens altas: elas permanecem no alvo e são auditadas separadamente. `repo_metadata_fetched` assumiu um único valor no subconjunto reutilizado do desenvolvimento, mas permaneceu no conjunto congelado de atributos. Como não discrimina essas observações, sua importância por permutação é zero; removê-lo após consultar o teste alteraria o pipeline avaliado. Essa escolha é registrada como limitação de pré-processamento.

## 3.6 Representação textual e seleção de atributos

O vocabulário textual é reconstruído em cada dobra. A exploração comparou pools de 5.000, 10.000 e 20.000 candidatos TF-IDF e limites de 50, 100, 200 e 500 termos. Para o alvo contínuo, a triagem univariada utiliza `f_regression` sobre `log1p(repository_count)`, ajustada somente no treino da dobra.

Esse procedimento é uma etapa de seleção de atributos, mas não constitui vazamento quando é ajustado exclusivamente com os dados de treinamento da dobra. A validação serve para comparar configurações sem permitir que os rótulos da dobra de validação influenciem a seleção. O pool histórico de 211 termos não participa do novo pipeline.

São comparadas três visões de atributos:

- **skill-only:** atributos estruturais e textuais da skill;
- **repository-only:** atributos contextuais do repositório;
- **hybrid:** combinação dos blocos da skill e do repositório.

A configuração de 10.000 candidatos e 500 termos foi usada na comparação final das três visões com Ridge. Não há avaliação externa dessa configuração textual.

## 3.7 Modelos avaliados

O protocolo utiliza regressão para prever a contagem de reutilização. Os seis regressores comparados são Ridge, KNN, árvore de decisão, Random Forest, Gradient Boosting e HistGradientBoosting, todos treinados sobre `log1p(repository_count)`. As previsões são transformadas de volta à escala original para avaliação.

O benchmark compara os seis vieses indutivos sob as mesmas dobras e atributos. O critério principal de seleção é o menor RMSLE médio na validação agrupada. Random Forest e Gradient Boosting foram levados ao teste externo para documentar também o compromisso entre RMSLE e MAE; os dois resultados são apresentados sem seleção retroativa pelo teste.

Para comparar todas as famílias nas 25.918 observações reutilizadas do desenvolvimento, o benchmark completo utiliza a visão `repository-only`. As visões híbrida e `skill-only` foram avaliadas com Ridge em experimento distinto. Esses resultados não constituem uma comparação das seis famílias sobre texto e estrutura.


## 3.8 Métricas e análise de erros

A métrica principal é o RMSLE, calculado como:

$$
\mathrm{RMSLE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}\left[\log(1+y_i)-\log(1+\hat{y}_i)\right]^2}.
$$

O RMSLE reduz a influência de valores extremos e avalia o erro na escala logarítmica. As métricas secundárias são MAE na escala original, MAE na escala logarítmica e correlação de Spearman. Previsões negativas após a transformação inversa são truncadas em zero para o cálculo das métricas. Uma referência constante prediz a mediana do alvo do desenvolvimento.

Além das médias agregadas, os erros são analisados por quantis do alvo. Na auditoria de validação do Ridge híbrido, os cortes são calculados no treino de cada dobra. Na auditoria descritiva do Random Forest no teste, os cortes são calculados em todo o desenvolvimento (P50=2, P75=4, P90=8 e P95=16). São reportados RMSLE, MAE, erro mediano assinado e proporção de subestimação por faixa.

## 3.9 Explicabilidade

A análise textual explora termos recorrentes e a robustez ao mascaramento de artefatos. Para o Random Forest `repository-only`, a importância por impureza do ajuste final é complementada por permutação de cada atributo bruto em três dobras agrupadas do desenvolvimento, com uma permutação por atributo e dobra. Um cenário de previsão varia `repo_stars` em uma amostra da primeira dobra, mantendo os outros atributos fixos. O cenário descreve a resposta do modelo e não estima efeito causal.

Termos correlacionados podem compartilhar atribuições; estrelas, forks e histórico também podem atuar como proxies da origem ou visibilidade. A estabilidade é avaliada pelas três dobras, sem extrapolar a conclusão para outras amostras. As explicações são associações dependentes do modelo, não regras causais de autoria ou evidência de qualidade.

## 3.10 Reprodutibilidade

Os manifestos registram a revisão dos dados, assinaturas da amostra e da divisão, sementes, configurações principais, cardinalidades e métricas. As tabelas por dobra e o código documentam o restante dos cálculos. O Apêndice A fornece os comandos e relaciona cada tabela e figura do relatório ao artefato que a originou. Isso permite repetir a validação e auditar os números apresentados, sujeito à disponibilidade do snapshot público e dos recursos computacionais necessários.

O pipeline também documenta as limitações conhecidas: o agrupamento por repositório representativo não elimina todas as relações entre repositórios, a replicação exata não captura cópias modificadas e o teste externo representa um snapshot específico do ecossistema GitHub. Essas limitações são incorporadas à análise de validade e à interpretação dos resultados.
