# Fase 0 - Protocolo de pesquisa

## Estado do protocolo

Versao: 0.5  
Status: Ridge híbrido selecionado provisoriamente; auditoria textual concluída e subestimação da cauda identificada.

## Pergunta de pesquisa

Quais caracteristicas observaveis de uma skill estao associadas a maior intensidade de
reutilizacao como artefato de engenharia de software?

A pesquisa deve distinguir caracteristicas da skill, contexto do repositorio e sinais textuais.
As conclusoes sao associativas e preditivas; nao serao apresentadas como efeitos causais.

## Unidade de analise e grupos

- Unidade de analise: uma skill identificada por `content_hash`.
- Grupo de dependencia: `representative_repo_id`.
- Nenhum grupo pode ser dividido entre treino, validacao e teste.
- O teste externo congelado permanece intocado ate a avaliacao final.

## Alvo definido

O alvo principal e `repository_count` entre skills com `reused = 1`. A pergunta operacional e:

> quais caracteristicas estao associadas a uma maior quantidade observada de repositorios que
> reutilizam uma skill?

O alvo sera modelado na escala de contagem e, como alternativa de treinamento, na escala
`log1p(repository_count)`. Modelos de contagem com sobredispersao, especialmente Binomial
Negativa quando disponiveis, serao considerados. A classificacao ordinal nao faz parte do
protocolo principal.

RMSLE e a metrica principal de selecao. MAE na escala original, MAE na escala logaritmica e
Spearman serao metricas secundarias. A escolha foi registrada depois da EDA, que encontrou forte
assimetria e razao variancia/media aproximadamente igual a 22.

## Regras de leakage

- `repository_count`, `reused`, `reuse_intensity` e `split_stratum` nunca entram como features.
- Identificadores e atributos derivados do resultado nao entram como features.
- TF-IDF, imputacao, escalonamento, codificacao e selecao de features sao ajustados somente no
  treino de cada dobra.
- Nenhuma decisao de modelo, hiperparametro, limite textual ou limiar usa o teste externo.
- Todos os algoritmos usam as mesmas divisoes e seeds dentro de uma comparacao.

## Representacoes candidatas

- `skill-only`: texto e atributos estruturais da skill;
- `repository-only`: atributos contextuais do repositorio;
- `hybrid`: combinacao dos dois blocos.

## Grid textual exploratorio

- candidatos TF-IDF: 5.000, 10.000 e 20.000;
- termos selecionados: 50, 100, 200 e 500;
- selecao e vocabulário refeitos dentro de cada dobra;
- para o alvo contínuo, a triagem univariada usa `f_regression` sobre `log1p(repository_count)`;
- o pool historico de 211 termos nao participa do novo pipeline.

## Avaliacao

As metricas da regressao de contagem sao:

- principal: RMSLE;
- secundarias: MAE na escala original, MAE em `log1p`, Spearman e deviance de contagem quando
  aplicavel.

## Explicabilidade

A explicabilidade sera aplicada depois da escolha da formulacao e do melhor modelo. Serao
avaliadas importancia global, explicacoes locais e estabilidade por dobra e bootstrap. Nenhuma
feature sera descrita como robusta apenas por aparecer em uma unica execucao.

## Reprodutibilidade

Cada experimento deve registrar:

- configuracao completa;
- seed;
- divisao dos grupos;
- assinatura do pre-processamento;
- origem dos dados;
- versoes do ambiente;
- artefatos gerados;
- metricas e tempo de execucao.

## Uso do teste externo na trilha da disciplina

O benchmark final da disciplina utilizou a particao externa atual para comparar os dois candidatos
selecionados (`repository-only`). Essa avaliacao e valida para o relatorio da disciplina, mas
consome o status de teste cego dessa particao. Portanto, os resultados nao devem ser reutilizados
como avaliacao final independente do artigo MSR. A trilha MSR devera reservar uma nova particao
externa ou uma coleta temporal independente antes de qualquer afirmacao de generalizacao final.

## Proximo gate

O grid diagnóstico foi seguido por uma confirmação completa da configuração selecionada
(10.000 candidatos, 500 termos) em 25.918 skills reutilizadas. O Ridge em `log1p` é o candidato
atual; Poisson, Tweedie e SGD foram inferiores ou instáveis. A explicabilidade inicial mostrou
termos estáveis relacionados a fragmentos editoriais. A remoção de datas, números, URLs e
identificadores alterou pouco o RMSLE (0,568 para 0,570 no híbrido). A auditoria por quantis,
contudo, encontrou subestimação em 100% dos casos acima do P95. O próximo gate é comparar pesos,
calibração e modelos de contagem sobredispersa antes do congelamento e do teste externo.


---

# Dicionario de dados

Versao de desenvolvimento: amostra oficial do GitSkills no commit
`fff3df93beeee51a5444450891751f143e38da3e`.

## Tabelas de origem

| Tabela | Linhas na amostra | Unidade |
|---|---:|---|
| `artifacts` | 29.786 | ocorrencia de um arquivo `SKILL.md` |
| `repos` | 11.841 | repositorio GitHub |
| `artifact_siblings` | 47.829 | entrada na pasta de uma skill representativa |
| `mining_runs` | 7 | execucao do processo de coleta |

## Tabela analitica `skills_target.parquet`

Cada linha representa um conteudo distinto, identificado pelo hash Git do arquivo.

| Campo | Origem | Descricao | Uso planejado |
|---|---|---|---|
| `content_hash` | `artifacts.file_sha` | hash exato do conteudo | identificador da instancia |
| `occurrence_count` | derivado | numero de ocorrencias do hash | auditoria, nunca feature |
| `repository_count` | derivado | repositorios distintos contendo o hash | construcao do alvo |
| `reused` | derivado | indica `repository_count >= 2` | alvo principal |
| `reuse_intensity` | derivado | 0: 2 repositorios; 1: 3-4; 2: 5-9; 3: 10+; ausente se `reused=0` | alvo condicional |
| `split_stratum` | derivado | cinco estratos usados para selecionar a divisao externa | controle experimental |
| `representative_repo_id` | `artifacts.repo_full_name` | repositorio da copia representativa | agrupamento e contexto |
| `representative_path` | `artifacts.path` | caminho da copia representativa | auditoria |
| `content` | `artifacts.content` | texto integral da copia representativa | features textuais |
| `frontmatter_valid` | `artifacts` | validade do YAML inicial | feature estrutural |
| `name`, `description` | `artifacts` | campos extraidos do front matter | features textuais |
| `body_chars` | `artifacts` | tamanho do corpo em caracteres | feature estrutural |
| `sibling_count`, `sibling_bytes` | `artifacts` | composicao da pasta representativa | features estruturais |
| `has_scripts`, `has_references` | `artifacts` | presenca de materiais auxiliares | features estruturais |
| `repo_stars`, `repo_language` | `repos` | contexto da copia representativa | experimento secundario |

`occurrence_count` e `repository_count` sao excluidos das features porque definem ou revelam o
alvo. A composicao da pasta e os metadados descrevem a copia representativa, nao uma origem
historica comprovada.

## Dados finais

`final_modeling_sample.parquet` contem 200 mil conteudos selecionados deterministicamente da
populacao completa. `final_targets.parquet` contem os dois alvos, a contagem usada para cria-los
e o estrato de divisao. `final_structured_features.parquet` contem os identificadores, o alvo
binario, a divisao treino/teste, os atributos estruturais e os atributos textuais agregados.
O alvo de intensidade deve ser unido pela chave `content_hash` apenas no momento da modelagem,
sem entrar na matriz de atributos. O texto integral nao e duplicado nessa segunda tabela e deve
ser lido da amostra de modelagem somente quando um experimento textual exigir.

`final_split_assignments.parquet` e a fonte autoritativa da divisao externa final. Ela contem
os identificadores, os dois alvos, o estrato e o split. Nenhum valor de
`representative_repo_id` pode ocorrer simultaneamente em treino e teste.
