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
