# Protocolo principal da pesquisa (v1.1)

**Estado:** definido em 10/11/2026, depois de uma fase exploratória no desenvolvimento; v1.1 registra o holdout construído, o tuning aninhado e os modelos congelados **antes** da avaliação do holdout (tag git `frozen-before-holdout`). Este documento fixa o que é **confirmatório** (hipóteses H1–H4, um único modelo final, um holdout novo) e o que é **exploratório**. Decisões tomadas depois de ver resultados estão marcadas como *dependentes dos dados* (seção 9).

## 1. Pergunta e posicionamento

Quais características observáveis de uma skill e do seu repositório estão associadas à difusão de seu conteúdo **entre donos** no GitHub, e quanto dessa previsibilidade é artefato da construção do dataset?

Duas contribuições, nesta ordem:

1. **Metodológica:** a escolha do representante no GitSkills (melhor classe de localização, depois ordem alfabética de repositório e caminho) torna `location_class` e `history_fetched` função do número de cópias; o alvo `repository_count` mistura difusão com autocópia; poucos donos concentram a amostra.
2. **Preditiva (modesta e testada com cuidado):** com atributos livres desses artefatos, há sinal para o reúso entre donos.

Os resultados são **associações observacionais**; não há afirmação causal nem de qualidade da skill.

## 2. Dados e população

- Fonte: GitSkills, revisão Hugging Face `ebab17454a7c236f8f26b183567f1a126f42e3f8` (coleta de julho de 2026; referência temporal fixa em 2026-07-20).
- Amostra: 200.000 conteúdos (`md5(content_hash || ':20261009')`); divisão externa original agrupada por repositório: 160.000 desenvolvimento / 40.000 teste.
- Unidade: conteúdo distinto de `SKILL.md` (`content_hash`).
- **População de estudo:** conteúdos com `repository_count >= 2` (reutilizados). Desenvolvimento: 25.918 conteúdos de 7.128 donos.
- **Limite por dono (principal):** no máximo **50 conteúdos por dono do repositório representativo**, escolhidos pela ordem de `content_hash` (determinística). Resulta em 17.733 conteúdos. Sensibilidades: sem limite, exclusão de donos com >200, >500 e >1000 linhas.
- A partição externa original **está consumida** (foi consultada e 17% de suas linhas têm donos presentes no desenvolvimento). Nenhuma afirmação confirmatória se apoia nela (seção 8).

## 3. Alvos

| Papel | Alvo | Definição |
|---|---|---|
| **Principal (classificação)** | `cross_owner` | `owner_count >= 2`, com `owner_count` = donos distintos que contêm o mesmo conteúdo (todas as ocorrências brutas) |
| Secundário (regressão) | `log1p(owner_count)` | métrica RMSLE |
| Sensibilidade | `log1p(repository_count)` | alvo antigo |

Justificativa: `repository_count` conta autocópia (28% das skills reutilizadas têm todas as cópias em um único dono) e é limite inferior da população (a busca do GitHub é parcial).

## 4. Atributos

**Principais (limpos, 52 colunas):**

| Bloco | n | Conteúdo |
|---|---:|---|
| `skill` | 34 | estrutura do SKILL.md (tamanho, títulos, listas, blocos de código, links, front matter, scripts/referências, pasta) e `filename` |
| `repo` | 8 | `log1p` de estrelas e forks, `repo_is_fork`, idade, recência do push, dias ativos, linguagem, licença |
| `catalog` | 2 | `log1p` do número de skills no repositório e indicador ≥10 |
| `path` | 8 | profundidade, raiz, `template`, `plugin`, `vendor`, `docs`, `pkg`, diretório de outra ferramenta |

**Atalhos (analisados como achado, fora dos modelos principais):** `location_class`, `history_fetched`, `commit_count`, `.claude` no caminho. Dependem da regra de escolha do representante (auditoria em `results/owner/audit_by_*`).

Nunca são preditores: `repository_count`, `owner_count`, `occurrence_count`, `canonical_share`, `any_canonical`, estratos e partição.

Pré-processamento (sempre ajustado no treino de cada dobra): mediana + indicador de ausência + escala (sem média) para numéricos; categoria constante + one-hot (`min_frequency=10`) para categóricos.

## 5. Modelos e validação

- **Modelo principal:** Random Forest (regressão: 100 árvores, `max_depth=12`, `min_samples_leaf=3`, `max_features=sqrt`; classificação: 300 árvores com os mesmos limites; `random_state=2027`).
- Referências lineares: Ridge (`alpha=10`) e regressão logística (`C=1`).
- Baselines: prevalência (classificação), média e mediana de `log1p` (regressão).
- **Validação:** 3 repetições × 3 dobras `StratifiedGroupKFold`, **agrupadas por dono**, estratificadas por faixa de `owner_count`; sementes 2027, 2028, 2029; mesmas partições para todos os modelos. Nenhum dono aparece em treino e validação de uma mesma dobra.
- **Sensibilidade ao `name`:** por dobra, fração das linhas de validação com `name` visto no treino e métricas no subconjunto sem sobreposição.
- **Tuning aninhado e seleção (executados):** `skillreuse final-model`; seis famílias (linear, KNN, árvore, Random Forest, Gradient Boosting, HistGradientBoosting) × {padrão, buscada} sobre `clean_full`; externa 2 repetições × 3 dobras agrupadas por dono; interna 3 dobras agrupadas por dono, `RandomizedSearchCV` com 6 candidatas; seleção pela maior média externa (AUC) ou menor (RMSLE). **Desvio por custo computacional:** o protocolo previa 3×3 e 12 candidatas.
- **Modelos congelados** (`results/owner/final/final_config_*.json`): classificação = Random Forest padrão (AUC aninhado 0,716, baseline 0,5); regressão = HistGradientBoosting padrão (RMSLE aninhado 0,6695, baseline 0,7197). Em ambas as tarefas as famílias empatam estatisticamente (IC das diferenças inclui zero); a regra pré-especificada seleciona a melhor média. A busca não superou a configuração padrão. Só esses dois modelos vão ao holdout. H2 foi testada com RF no desenvolvimento; o modelo final de regressão é o HistGB selecionado pela regra.

## 6. Métricas

- Principais: **AUC-ROC** (classificação) e **RMSLE** (regressão).
- Secundárias: AUC-PR, Brier, log-loss, MAE, MAE em `log1p`, Spearman.
- Incerteza: diferença pareada por (repetição, dobra), IC95% e **teste t com correção de Nadeau–Bengio** (as repetições reutilizam os mesmos dados). O Wilcoxon pareado é complementar, não decide.

## 7. Hipóteses pré-especificadas (família com correção de Holm, α = 0,05)

| ID | Hipótese (contra a alternativa de ausência de efeito) | Tarefa | Comparação |
|---|---|---|---|
| H1 | os atributos limpos preveem reúso entre donos | classificação | AUC `clean_full` RF contra prevalência (0,5) |
| H2 | os atributos limpos reduzem o RMSLE de `owner_count` | regressão | RF `clean_full` contra média de `log1p` |
| H3a | a skill acrescenta ao repositório e catálogo | classificação | `clean_full` contra `repo+catalog` (AUC) |
| H3b | repositório e catálogo acrescentam à skill | classificação | `clean_full` contra `skill` (AUC) |
| H4 | os atalhos acrescentam aos atributos limpos (achado metodológico) | regressão | `clean_full+shortcuts` contra `clean_full` (RMSLE) |

Todo o resto é **exploratório**: reportar IC e p nominal, sem declarar significância.

Executar: `make primary` → `results/owner/primary/primary_hypotheses.csv`.

## 8. Confirmação em holdout novo

A fase acima é **exploratória para fins confirmatórios**, pois os resultados foram vistos antes de fixar o protocolo. A confirmação exige um holdout nunca consultado:

- Conteúdos fora dos 200.000 amostrados (≈1,68 milhão) com `repository_count >= 2`, **donos disjuntos** dos donos de todas as skills reutilizadas da amostra (desenvolvimento e teste original; critério mais rígido que o inicial), com o mesmo limite de 50 conteúdos por dono. Construído por `skillreuse holdout`: 25.011 conteúdos de 10.484 donos, sobreposição zero de donos e de conteúdos, máximo de 37 conteúdos por dono, semente `20261110`. A prevalência de `cross_owner` (80,8%) foi vista como verificação de integridade; nenhum modelo foi avaliado antes do congelamento.
- **Deslocamento de composição (limitação):** o holdout tem 36% de skills `canonical` e 9% `other` (desenvolvimento: 21% e 21%), porque os donos concentrados ficam de fora.
- Mesmas definições de alvos e atributos, mesmo código, **um único** modelo final (o selecionado na seção 5), uma única avaliação.
- Estimativa com **bootstrap por conglomerado (dono)**, 2.000 reamostras. Critério de sucesso: limite inferior do IC95% do AUC > 0,5 e limite superior do IC95% da diferença de RMSLE para a média < 0.
- Se o critério falhar, o resultado é reportado como tal.

## 9. Decisões dependentes dos dados (transparência)

1. O **limite de 50 por dono** foi escolhido depois de ver que o AUC varia de 0,615 a 0,747 conforme o limiar de exclusão (poucos donos com 500–1000 linhas dominam o resultado). Escolher outro valor altera as estimativas; a sensibilidade (25/50/100, sem limite, limiares) é obrigatória no relatório.
2. A escolha de `cross_owner` como alvo principal e de `clean_full` como visão principal veio da fase exploratória.
3. Os atalhos foram excluídos depois de a auditoria mostrar que a probabilidade de o representante ser `canonical` sobe de 14% (2 repositórios) para 92% (>50) enquanto a fração real de cópias em `canonical` fica entre 5% e 13%.

## 10. Interpretação e limites

- Importância por permutação em dobras agrupadas por dono (`skillreuse owner-importance`); correlação entre atributos limita a leitura individual.
- Os atributos descrevem **uma cópia** (o representante) de cada conteúdo; agregados sobre todas as cópias dependeriam das mesmas ocorrências que definem o alvo (vazamento) e não são usados.
- Cópia exata não captura versões modificadas; `owner_count` é limite inferior; o snapshot é de julho de 2026.
- A atribuição de dono usa a conta do repositório; organizações e contas de uma mesma pessoa podem ser distintas.

## 11. Dicionário de dados

Ver `results/owner/` (EDA, auditoria) e `results/owner/primary/manifest.json`. Colunas originais: tabela `artifacts` e `repos` do GitSkills (arXiv:2608.10906).
