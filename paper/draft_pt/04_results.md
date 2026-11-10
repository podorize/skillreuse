# 4. Resultados

## 4.1 O alvo e a concentração por donos

No desenvolvimento (25.918 conteúdos reutilizados de 7.128 donos), o número de donos tem distribuição semelhante à de repositórios: forte assimetria positiva e sobredispersão.

| | média | P50 | P75 | P90 | P95 | P99 | máx. | variância/média |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `repository_count` | 5,08 | 2 | 4 | 8 | 16 | 55 | 390 | 22,1 |
| `owner_count` | 4,50 | 2 | 3 | 8 | 15 | 53 | 379 | 23,1 |

: Distribuição dos alvos de contagem no desenvolvimento.

A diferença está na base da distribuição: **28,0%** dos conteúdos reutilizados têm `owner_count = 1`, isto é, todas as cópias estão em um único dono. A autocópia retira 11% da massa total de `repository_count`. Poucos donos concentram a amostra: o maior responde por 11,9% dos conteúdos, os cinco maiores por 24,5%.

Os cinco maiores donos respondem por 11,9%, 4,7%, 3,5%, 2,2% e 2,2% dos conteúdos (identificadores omitidos). O maior e o quarto são casos claros de autocópia: seus conteúdos têm em média 2,0 repositórios, mas só 1,0 dono. Além disso, 17,2% dos conteúdos não têm `name` no front matter, e 20,1% (24,3% dos que têm `name`) compartilham o `name` com outro conteúdo, indício de variantes da mesma skill.

## 4.2 O viés de seleção do representante

A tabela a seguir compara, na população completa (1.877.981 conteúdos), a fração de conteúdos cujo representante é `canonical` com a fração das cópias que de fato estão em `canonical`.

| Repositórios | conteúdos | representante é `canonical` | cópias em `canonical` |
|---|---:|---:|---:|
| 1 | 1.571.893 | 13,3% | 12,6% |
| 2 | 172.624 | 13,8% | 8,5% |
| 3 | 49.535 | 19,5% | 9,1% |
| 4–5 | 36.312 | 24,1% | 10,9% |
| 6–10 | 24.142 | 30,4% | 10,1% |
| 11–50 | 19.589 | 57,7% | 7,7% |
| mais de 50 | 3.886 | 91,7% | 5,1% |

: Efeito da regra do representante, por número de repositórios (população completa).

A probabilidade de o representante ser `canonical` cresce de 14% a 92% com o número de cópias, enquanto a fração real de cópias nessa pasta é plana ou decrescente (5–13%). Isso é consequência da regra recuperada (Seção 3.3): basta uma cópia em `.claude/skills/` para o representante ser `canonical`. A classe de localização do representante, e o histórico de commits que só existe para locais padrão, **codificam o número de cópias**.

![Distribuições de repositórios e donos (esquerda) e efeito da regra do representante (direita)](../../results/owner/eda_owner_and_selection.png)

Na primeira versão do estudo, com `repository_count` como alvo e atributos de repositório, isso inflava o resultado. Com o Random Forest e o desenvolvimento completo, os atributos de repositório davam RMSLE 0,575; sem `history_fetched` e `commit_count`, 0,604 (diferença de +0,029, IC95% [0,018; 0,040]), contra 0,623 da baseline de média de `log1p`. Só a localização do representante (três categorias) dava 0,584, e o histórico sozinho 0,594. A visão "de repositório" era, em boa parte, uma proxy da localização.

## 4.3 Protocolo principal

Com os atalhos fora, 17.733 conteúdos (limite de 50 por dono) e validação agrupada por dono, os resultados do Random Forest (RF) e dos modelos lineares são:

| Visão | AUC RF | AUC logística | RMSLE RF | RMSLE Ridge |
|---|---:|---:|---:|---:|
| skill | 0,660 ± 0,022 | 0,614 | 0,698 ± 0,007 | 0,713 |
| repositório | 0,650 ± 0,019 | 0,655 | 0,707 ± 0,011 | 0,702 |
| repositório + catálogo | 0,666 ± 0,009 | 0,662 | 0,696 ± 0,010 | 0,699 |
| caminho + catálogo | 0,579 ± 0,019 | 0,576 | 0,718 ± 0,018 | 0,714 |
| **limpo completo** | **0,716 ± 0,024** | 0,680 | **0,679 ± 0,011** | 0,694 |
| limpo + atalhos | 0,717 ± 0,022 | 0,682 | 0,637 ± 0,015 | 0,664 |
| só atalhos | 0,500 ± 0,015 | 0,510 | 0,688 ± 0,010 | 0,693 |
| *baseline* | 0,500 | 0,500 | 0,720 (média), 0,780 (mediana) | |

: Desempenho por visão (média ± desvio de 9 dobras agrupadas por dono).

Todas as hipóteses pré-especificadas são rejeitadas após a correção de Holm:

| | Comparação | Resultado | IC95% da diferença | p (Holm) |
|---|---|---|---|---:|
| H1 | AUC do RF limpo contra 0,5 | 0,716 | +0,216 [0,173; 0,259] | < 0,001 |
| H2 | RMSLE do RF limpo contra a média | 0,679 contra 0,720 | −0,041 [−0,049; −0,033] | < 0,001 |
| H3a | limpo contra repositório+catálogo (AUC) | 0,716 contra 0,666 | +0,050 [0,017; 0,083] | 0,008 |
| H3b | limpo contra skill (AUC) | 0,716 contra 0,660 | +0,056 [0,040; 0,072] | < 0,001 |
| H4 | limpo+atalhos contra limpo (RMSLE) | 0,637 contra 0,679 | −0,042 [−0,055; −0,029] | < 0,001 |

: Hipóteses pré-especificadas no desenvolvimento (exploratório até a confirmação da Seção 4.7).

Há sinal real, mas modesto: o AUC de 0,72 e uma redução de 6% do RMSLE. Skill e repositório contribuem de forma **complementar**: cada bloco isolado fica em AUC 0,65–0,67, e a combinação em 0,72 (H3a e H3b). Os atalhos acrescentam à regressão (H4), mas, sozinhos, não preveem o reúso entre donos (AUC 0,50): o ganho de regressão vem da correlação deles com o número de cópias, não do reúso entre donos. O caminho, isolado, quase não prevê.

## 4.4 Robustez

**Limite por dono.** O desempenho do RF limpo muda pouco no RMSLE, mas muito no AUC, conforme o tratamento dos donos concentrados:

| Tratamento | conteúdos | AUC RF limpo | ganho de RMSLE (limpo) | ganho (só atalhos) |
|---|---:|---:|---:|---:|
| sem limite | 25.918 | 0,747 ± 0,140 | 0,039 | 0,040 |
| excluir donos com mais de 1000 | 21.615 | 0,615 ± 0,036 | 0,036 | 0,040 |
| excluir donos com mais de 500 | 19.567 | 0,708 ± 0,037 | 0,030 | 0,041 |
| excluir donos com mais de 200 | 18.443 | 0,732 ± 0,035 | 0,037 | 0,040 |
| **50 por dono** | 17.733 | 0,716 ± 0,024 | 0,041 | 0,031 |

: Sensibilidade ao tratamento de donos concentrados (RF, baseline = média de `log1p`).

O RMSLE é estável (ganho de 0,03–0,04), mas o AUC oscila de 0,615 a 0,747: poucos donos entre 500 e 1000 conteúdos derrubam o AUC em 0,1, e sem limite uma única dobra concentra um dono e o desvio chega a 0,14. O desempenho depende, portanto, da composição de uma dúzia de contas. O limite de 50 estabiliza as estimativas, mas é uma escolha dependente dos dados.

**Outras verificações.** (i) Fixando a classe de localização e treinando dentro de cada uma (donos com mais de 500 conteúdos excluídos), o RF limpo ainda supera a baseline: `canonical` (5.439 conteúdos) RMSLE −0,141 e AUC 0,714; `skills-dir` (10.721) −0,013 e 0,672; `other` (3.810) −0,011 e 0,828. O sinal não é só localização. (ii) Com `repository_count` como alvo, o ganho do RF limpo é igual (−0,031 contra −0,030). (iii) Só 18,8% das linhas de validação têm `name` visto no treino; no subconjunto sem sobreposição o ganho persiste (RMSLE −0,039 contra −0,041 em todas; AUC 0,726 contra 0,716), o que indica que variantes do mesmo `name` não explicam o resultado. (iv) A validação agrupada por repositório, em vez de por dono, não produziu estimativas maiores (RMSLE 0,657 contra 0,655; AUC 0,735 contra 0,747, com o desvio de 0,14 do caso sem limite).

## 4.5 Interpretação

![Importância por permutação do RF limpo (dobras agrupadas por dono; donos com mais de 500 conteúdos excluídos)](../../results/owner/figures/permutation_importance.png)

Para o reúso entre donos, o bloco de repositório pesa mais (queda de 0,104 no AUC ao permutar) que o de skill (0,084), catálogo (0,016) e caminho (≈ 0). Entre as colunas, domina o `log_stars` (0,056), seguido de `log_forks` (0,021) e recência do último push (0,016): skills em repositórios mais visíveis são mais copiadas por outros donos. Na regressão, repositório (0,019) e skill (0,018) pesam parecido, e a maior coluna é a licença (0,013), seguida do tamanho do catálogo (0,011) e da profundidade do caminho (0,005). Não identificamos por que a licença pesa tanto; pode ser um indicador de repositórios que clonam catálogos. As incertezas são grandes (6 dobras), e a correlação entre atributos limita a leitura individual.

## 4.6 Seleção de modelo e tuning

| Família | AUC padrão | AUC buscado | RMSLE padrão | RMSLE buscado |
|---|---:|---:|---:|---:|
| linear | 0,678 | 0,689 | 0,692 | 0,690 |
| KNN | 0,667 | 0,690 | 0,703 | 0,686 |
| árvore | 0,614 | 0,634 | 0,807 | 0,699 |
| Random Forest | **0,716** | 0,701 | 0,679 | 0,675 |
| Gradient Boosting | 0,691 | 0,704 | 0,690 | 0,671 |
| HistGradientBoosting | 0,693 | 0,682 | **0,670** | 0,688 |
| *baseline* | 0,500 | | 0,720 | |

: Validação aninhada (média de 6 dobras externas agrupadas por dono; 6 candidatas por busca).

Pela regra pré-especificada, congelamos o RF padrão (classificação) e o HistGradientBoosting padrão (regressão). **As famílias empatam:** o RF contra o segundo colocado (Gradient Boosting buscado) tem diferença de AUC de 0,012 (IC95% [−0,003; 0,026]), e na regressão o HistGB contra o Gradient Boosting buscado tem −0,001 [−0,027; 0,024]. Só a árvore de decisão (e, com menos evidência, o KNN padrão) fica atrás, e na classificação nem essas diferenças resistem à correção de Holm (p entre 0,06 e 0,13). A busca de hiperparâmetros não superou a configuração padrão (RF classificação: padrão 0,716 contra buscado 0,701).

## 4.7 Confirmação no holdout

Os modelos congelados foram avaliados uma vez em 25.011 conteúdos de 10.484 donos disjuntos do desenvolvimento.

| Tarefa | Modelo | Resultado | Baseline | IC95% (bootstrap por dono) |
|---|---|---:|---:|---|
| `cross_owner` | Random Forest | AUC **0,731**; AUC-PR 0,913; Brier 0,140; log-loss 0,444 | 0,5; 0,808; 0,155; 0,490 | AUC [0,716; 0,745]; AUC-PR [0,905; 0,921] |
| `log1p(owner_count)` | HistGradientBoosting | RMSLE **0,625**; Spearman 0,330 | 0,664 (média de `log1p`) | diferença −0,038 [−0,044; −0,032] |

: Avaliação única no holdout.

![Curva ROC e calibração do RF no holdout](../../results/owner/figures/holdout_roc_calibration.png)

Os critérios de sucesso foram atendidos: o limite inferior do AUC está acima de 0,5 e o limite superior da diferença de RMSLE está abaixo de zero. O AUC do holdout (0,731) é compatível com a estimativa aninhada do desenvolvimento (0,716). O holdout tem composição diferente (36% de skills `canonical` contra 21% no desenvolvimento, e 9% `other` contra 21%), pois os donos concentrados ficam de fora por construção. H3 e H4 não foram reavaliadas no holdout e permanecem exploratórias.
