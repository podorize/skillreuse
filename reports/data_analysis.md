# Analise exploratoria dos dados do SkillReuse

## Escopo e protocolo

A analise segue o roteiro do Capitulo 2 - Analise fornecido para o projeto: caracterizacao dos dados, classificacao dos atributos por tipo e escala, estatistica descritiva univariada e analise multivariada por correlacoes e visualizacoes.

As estatisticas da populacao e o balanceamento da amostra usam todos os dados. Comparacoes entre atributos preditivos e o alvo usam somente o treino congelado, preservando o teste para a avaliacao final.

## Caracterizacao

- Populacao: 1.877.981 conteudos distintos e 3.797.117 ocorrencias.
- Amostra de modelagem: 200.000 conteudos; treino: 160.000.
- Alvo positivo na amostra: 16.20%.
- Estrutura: dados mistos; texto nao estruturado transformado em atributos estruturados, categorias, contagens e metadados.
- Tempo: snapshot estatico da coleta; a ordem temporal interna dos textos e relevante, mas a tarefa atual nao e uma previsao temporal.
- Dependencia: conteudos podem compartilhar repositorios; por isso a divisao e agrupada por repositorio representativo.
- Volume: a populacao e processada fora da memoria; a amostra de 200 mil conteudos e processada em lotes.

## Qualidade e valores ausentes

| feature | missing_count | missing_percentage |
| --- | --- | --- |
| commit_count | 151496 | 75.7480 |
| first_commit_at | 151496 | 75.7480 |
| last_commit_at | 151496 | 75.7480 |
| repo_license | 101283 | 50.6415 |
| repo_language | 17217 | 8.6085 |
| composition_truncated | 149 | 0.0745 |
| has_references | 149 | 0.0745 |
| has_scripts | 149 | 0.0745 |
| sibling_bytes | 149 | 0.0745 |
| sibling_count | 149 | 0.0745 |

O historico de commits tem a maior ausencia. Essa ausencia representa principalmente indisponibilidade de coleta, nao o valor zero; deve ser modelada com indicador de ausencia e avaliada por ablacao.

Atributos sem variacao observada devem ser removidos do conjunto preditivo:

| feature | distinct_non_missing_values | only_value | missing_count |
| --- | --- | --- | --- |
| repo_is_fork | 1 | 0.0000 | 69 |
| repo_metadata_fetched | 1 | 1.0000 | 0 |

## Estatistica univariada

Os atributos de contagem apresentam forte assimetria positiva e caudas longas. Por isso, mediana e IQR sao mais informativos que media e desvio-padrao isoladamente, e transformacoes como log(1+x) sao candidatas naturais para modelos lineares.

| feature | median | q1 | q3 | skewness | outlier_percentage_1_5_iqr |
| --- | --- | --- | --- | --- | --- |
| markdown_link_count | 0.0000 | 0.0000 | 0.0000 | 77.3555 | 20.4125 |
| markdown_blockquote_count | 0.0000 | 0.0000 | 0.0000 | 22.6842 | 18.5475 |
| sibling_bytes | 0.0000 | 0.0000 | 3890.0000 | 140.5566 | 18.4588 |
| sibling_count | 0.0000 | 0.0000 | 1.0000 | 112.9390 | 17.8821 |
| repo_stars | 1.0000 | 0.0000 | 7.0000 | 40.7034 | 16.6941 |
| markdown_url_count | 0.0000 | 0.0000 | 1.0000 | 83.5172 | 14.5875 |
| repo_forks | 0.0000 | 0.0000 | 2.0000 | 64.9635 | 14.0439 |
| markdown_checkbox_count | 0.0000 | 0.0000 | 0.0000 | 11.1079 | 12.5256 |
| commit_count | 1.0000 | 1.0000 | 2.0000 | 22.8325 | 12.3197 |
| text_average_sentence_words | 21.3612 | 15.6667 | 35.0000 | 29.7204 | 9.8625 |

## Relacoes com o alvo no treino

A tabela abaixo ordena os atributos numericos pela magnitude da correlacao ponto-bisserial. Esses valores descrevem associacao marginal, nao causalidade e nem desempenho preditivo fora da amostra.

| feature | negative_median | positive_median | standardized_mean_difference | point_biserial_correlation |
| --- | --- | --- | --- | --- |
| text_average_word_length | 5.6374 | 5.8303 | 0.2600 | 0.0954 |
| commit_count | 1.0000 | 1.0000 | -0.1851 | -0.0740 |
| markdown_heading_count | 11.0000 | 13.0000 | 0.1703 | 0.0626 |
| text_nonempty_line_count | 88.0000 | 99.0000 | 0.1461 | 0.0538 |
| text_line_count | 117.0000 | 132.0000 | 0.1417 | 0.0521 |
| markdown_ordered_list_items | 5.0000 | 6.0000 | 0.1411 | 0.0519 |
| markdown_fenced_code_blocks | 2.0000 | 2.0000 | 0.1302 | 0.0479 |
| text_lexical_diversity | 0.4815 | 0.4702 | -0.1229 | -0.0452 |
| markdown_inline_code_count | 12.0000 | 8.0000 | -0.1073 | -0.0395 |
| markdown_unordered_list_items | 18.0000 | 25.0000 | 0.1024 | 0.0377 |
| markdown_checkbox_count | 0.0000 | 0.0000 | 0.0799 | 0.0294 |
| frontmatter_name_char_count | 15.0000 | 15.0000 | -0.0785 | -0.0289 |

Taxa de reutilizacao por localizacao do arquivo:

| value | count | positive_rate |
| --- | --- | --- |
| skills-dir | 89690 | 0.1664 |
| other | 47141 | 0.1178 |
| canonical | 23169 | 0.2348 |

Taxas de categorias com poucas observacoes sao instaveis e nao devem ser interpretadas isoladamente. A associacao de `commit_count` tambem exige cautela, pois o historico foi coletado para apenas cerca de um quarto dos conteudos.

## Dependencia entre atributos

As maiores correlacoes absolutas de Spearman indicam grupos redundantes de medidas de tamanho textual, estrutura Markdown e popularidade do repositorio. Isso recomenda regularizacao nos modelos lineares e analises de ablacao por grupos de atributos.

| feature_1 | feature_2 | correlation |
| --- | --- | --- |
| text_word_count | text_unique_word_count | 0.9707 |
| text_word_count | text_lexical_diversity | -0.8497 |
| text_unique_word_count | text_sentence_count | 0.7865 |
| text_word_count | text_sentence_count | 0.7789 |
| repo_stars | repo_forks | 0.7719 |
| text_word_count | markdown_heading_count | 0.7382 |
| text_unique_word_count | markdown_heading_count | 0.7189 |
| text_unique_word_count | text_lexical_diversity | -0.7110 |
| text_unique_word_count | markdown_unordered_list_items | 0.6760 |
| markdown_heading_count | markdown_fenced_code_blocks | 0.6652 |
| text_word_count | markdown_unordered_list_items | 0.6418 |
| text_lexical_diversity | markdown_heading_count | -0.6270 |

## Implicacoes para a modelagem

1. Preservar a prevalencia natural no teste e comparar pesos de classe apenas no treino.
2. Aplicar transformacoes e imputacao dentro de cada dobra de validacao cruzada.
3. Usar log(1+x) ou escalonamento robusto para contagens muito assimetricas.
4. Representar ausencias com indicadores quando significarem falha ou indisponibilidade de coleta.
5. Separar experimentos textuais, estruturais e contextuais para medir o ganho de cada grupo.
6. Nao interpretar reutilizacao como qualidade, origem historica ou relacao causal.

## Figuras produzidas

- `analysis_missingness.png`: cobertura dos atributos.
- `analysis_numeric_distributions.png`: histogramas por classe no treino.
- `analysis_boxplots.png`: quartis e dispersao por classe.
- `analysis_correlation_heatmap.png`: correlacao multivariada.
- `analysis_categorical_reuse_rates.png`: frequencias condicionais categoricas.
- `analysis_scatter_text_structure.png`: relacao bivariada entre texto e estrutura.
