# 3. Metodologia

## 3.1 Desenho da pesquisa

O estudo é empírico, quantitativo e preditivo, com uma camada interpretativa; seu objetivo é medir informação preditiva, não estimar efeitos causais \cite{shmueli2010explain}. A fase de **desenvolvimento** (exploratória) inclui as auditorias do dataset, a definição dos alvos e das visões de atributos, a validação agrupada e a seleção de modelo. A fase de **confirmação** avalia, uma única vez, os modelos congelados em um holdout novo, com critérios de sucesso fixados antes. O protocolo completo, inclusive as decisões tomadas depois de ver resultados, está em `paper/protocol.md`.

## 3.2 Dados e unidade de análise

A fonte é o corpus GitSkills \cite{gitskills2027}, revisão `ebab17454a7c236f8f26b183567f1a126f42e3f8`: 3.797.117 arquivos `SKILL.md` de 282.200 repositórios (julho de 2026), agrupados em 1.877.981 conteúdos distintos. A unidade é o conteúdo (`content_hash`). Todas as ocorrências (repositório e caminho) são mantidas, mas somente um **representante** por conteúdo recebe texto completo, metadados do repositório, pasta e histórico.

O desenvolvimento usa 200.000 conteúdos escolhidos por `md5(content_hash || ':20261009')`, divididos por repositório em 160.000 de desenvolvimento e 40.000 de teste. Condicionamos a `repository_count >= 2`: 25.918 conteúdos de 7.128 donos no desenvolvimento. Na população completa há 306.088 conteúdos nessa condição, dos quais 230.355 sob dois ou mais donos. A partição de teste original foi consultada na primeira versão do estudo e não sustenta afirmações confirmatórias. A data de referência para idade e recência é 20/07/2026.

## 3.3 Regra do representante e auditoria

A documentação do GitSkills indica que o representante prefere `.claude/skills/`, com desempate determinístico \cite{gitskills2027}. Recuperamos a regra a partir de todas as ocorrências brutas: ordenar por classe de localização (`canonical`, `skills-dir`, `other`) e, no empate, pelo menor par `(repo_full_name, path)`. Ela reproduz o representante em 97,2% dos 306.088 conteúdos com ao menos dois repositórios, e a classe do representante coincide com a melhor classe disponível em 100% deles. A auditoria compara, por faixa de número de repositórios, a fração de conteúdos cujo representante é `canonical` com a fração das cópias que de fato estão nessa classe, para saber se `location_class` e `history_fetched` (histórico só coletado para locais padrão e uma amostra) podem ser preditores legítimos.

## 3.4 Alvos

| Papel | Alvo | Definição |
|---|---|---|
| Principal (classificação) | `cross_owner` | `owner_count >= 2` |
| Secundário (regressão) | `log1p(owner_count)` | donos distintos entre todas as ocorrências |
| Sensibilidade | `log1p(repository_count)` | alvo da primeira versão |

: Alvos. O dono é o prefixo de `repo_full_name`.

Contar donos remove a autocópia entre repositórios de uma mesma conta. As contagens são limites inferiores, pois a busca de código do GitHub indexa apenas a branch padrão, arquivos de até 384 KB e repositórios recentes \cite{gitskills2027}.

## 3.5 Atributos

| Bloco | n | Conteúdo |
|---|---:|---|
| `skill` | 34 | estrutura do `SKILL.md` (tamanho, títulos, listas, código, links, front matter, scripts, pasta) e nome do arquivo |
| `repo` | 8 | `log1p` de estrelas e forks, fork, idade, recência do push, dias ativos, linguagem, licença |
| `catalog` | 2 | `log1p` do número de skills no repositório e indicador de pelo menos 10 |
| `path` | 8 | profundidade, raiz, `templates`, `plugins`, `vendor`, `docs`, `packages/src`, diretório de outra ferramenta |
| *atalhos* | 4 | `location_class`, `history_fetched`, `commit_count`, `.claude` no caminho |

: Blocos de atributos. Os modelos principais usam os quatro primeiros (52 colunas, visão *limpa*); os atalhos entram só como achado.

Os atalhos dependem da regra do representante e ficam fora dos modelos principais. Os atributos descrevem o representante, não todas as cópias; agregá-los sobre todas as cópias dependeria das mesmas ocorrências que definem o alvo e foi evitado. Imputação (mediana com indicador de ausência), escala e codificação (one-hot, `min_frequency=10`) são ajustadas dentro de cada dobra.

## 3.6 Validação e limite por dono

A validação é agrupada pelo **dono do repositório representativo**: 3 repetições de 3 dobras (`StratifiedGroupKFold`, sementes 2027–2029, estratos por faixa de `owner_count`), com as mesmas partições para todos os modelos. Como poucas contas concentram a amostra, aplicamos um **limite de 50 conteúdos por dono** (ordem de `content_hash`), restando 17.733 conteúdos. O valor 50 foi escolhido *depois* de observar a sensibilidade do AUC ao tratamento de donos concentrados (Seção 4.4); é, portanto, uma decisão dependente dos dados. Por dobra, medimos também a fração de linhas de validação cujo `name` do front matter aparece no treino e as métricas no subconjunto sem sobreposição (um agrupamento conjunto por dono e `name` forma um componente com 64% das linhas e foi descartado).

## 3.7 Modelos, baselines e tuning

O modelo principal é o Random Forest (100 árvores em regressão, 300 em classificação; `max_depth=12`, `min_samples_leaf=3`, `max_features=sqrt`). As referências lineares são Ridge ($\alpha=10$) e regressão logística ($C=1$); as baselines são a prevalência e a média e a mediana de `log1p`. O tuning é aninhado: seis famílias (linear, KNN, árvore, Random Forest, Gradient Boosting, HistGradientBoosting) em duas variantes (hiperparâmetros padrão e buscados por `RandomizedSearchCV`) sobre a visão limpa. A validação externa usa 2 repetições de 3 dobras agrupadas por dono; a busca interna, 3 dobras agrupadas por dono e 6 candidatas. A combinação com melhor média externa é congelada (se for "buscada", a busca é repetida em todo o desenvolvimento). O protocolo previa 3 repetições e 12 candidatas; reduzimos por custo computacional.

## 3.8 Métricas, inferência e interpretação

As métricas principais são o AUC-ROC (classificação) e o RMSLE (regressão); as secundárias, AUC-PR, Brier, log-loss, MAE e Spearman. As comparações são pareadas por (repetição, dobra), com IC95% e teste t com **correção de Nadeau–Bengio** para amostras repetidas; o Wilcoxon pareado é complementar. Cinco hipóteses pré-especificadas formam uma família corrigida por Holm ($\alpha=0{,}05$); as demais comparações são exploratórias.

| ID | Hipótese | Comparação |
|---|---|---|
| H1 | os atributos limpos preveem reúso entre donos | AUC do RF limpo contra 0,5 |
| H2 | os atributos limpos reduzem o RMSLE de `owner_count` | RF limpo contra a média de `log1p` |
| H3a | a skill acrescenta a repositório e catálogo | limpo contra repositório+catálogo (AUC) |
| H3b | repositório e catálogo acrescentam à skill | limpo contra skill (AUC) |
| H4 | os atalhos acrescentam aos atributos limpos | limpo+atalhos contra limpo (RMSLE) |

: Hipóteses pré-especificadas.

A interpretação usa importância por permutação do RF limpo em dobras agrupadas por dono, por coluna e por bloco (colunas do bloco permutadas em conjunto). É associativa; atributos correlacionados compartilham importância.

## 3.9 Holdout novo

Como o protocolo foi fixado após resultados exploratórios, a confirmação exige dados nunca consultados. O holdout reúne conteúdos fora dos 200.000 amostrados, com `repository_count >= 2`, cujos donos não aparecem entre as skills reutilizadas da amostra (desenvolvimento e teste original), com limite de 50 por dono: 25.011 conteúdos de 10.484 donos, sem donos nem conteúdos em comum com a amostra e com no máximo 37 conteúdos por dono. Viu-se apenas a prevalência de `cross_owner` (80,8%) como verificação de integridade. Os modelos escolhidos são treinados no desenvolvimento inteiro e avaliados **uma única vez**, com IC por bootstrap de donos (2.000 reamostras). O sucesso exige limite inferior do AUC acima de 0,5 e limite superior da diferença de RMSLE para a baseline abaixo de zero; uma trava impede reavaliações sem declaração explícita.
