# Metodologia

## Perguntas de pesquisa

Em que medida caracteristicas textuais, estruturais e contextuais permitem distinguir conteudos
de `SKILL.md` observados em mais de um repositorio daqueles observados em apenas um? Entre os
conteudos reutilizados, em que medida esses atributos permitem distinguir a intensidade da
difusao observada?

## Unidade de analise

Um conteudo distinto de `SKILL.md`, identificado por `content_hash`.

## Alvos

`reused = 1` quando o mesmo hash aparece em pelo menos dois repositorios distintos; caso
contrario, `reused = 0`.

O segundo alvo, `reuse_intensity`, e condicional a `reused = 1`:

| Codigo | Faixa de `repository_count` | Estrato de divisao |
|---:|---:|---|
| ausente | 1 | `not_reused` |
| 0 | 2 | `reuse_2` |
| 1 | 3-4 | `reuse_3_4` |
| 2 | 5-9 | `reuse_5_9` |
| 3 | 10 ou mais | `reuse_10_plus` |

A intensidade ausente representa a inaplicabilidade do segundo alvo, nunca uma quinta classe
condicional. `repository_count` e `occurrence_count` nao sao atributos preditivos. O primeiro
e mantido somente na tabela de alvos para auditoria.

## Principios metodologicos

- Deduplicar por `content_hash` antes da divisao dos dados.
- Ajustar transformacoes somente com os dados de treinamento.
- Preservar a prevalencia natural no teste final.
- Registrar versao dos dados, seeds, configuracoes e resultados.
- Interpretar reutilizacao como associacao observada, nao como qualidade ou causalidade.

## Divisao da amostra de desenvolvimento

A separacao da amostra oficial de desenvolvimento foi construida com
`StratifiedGroupKFold`, usando
`representative_repo_id` como grupo. Entre as cinco particoes candidatas, o pipeline escolhe de
forma deterministica a que mais se aproxima de 20% das linhas e da prevalencia global do alvo.
Nenhum repositorio representativo pode aparecer simultaneamente no treino e no teste.

Essa divisao e usada somente para desenvolver o pipeline; a avaliacao final usa a divisao da
amostra de 200 mil conteudos descrita abaixo.

## Revisao da divisao externa para os dois alvos

A divisao binaria original, com assinatura
`4ddc4b1b69302ef696d56c855ebc9d226bacc4599c56f5572f5bffba058bd8ae`,
equilibrava `reused`, mas as quatro faixas positivas diferiam substancialmente entre treino e
teste. Essa limitacao foi observada antes de qualquer treinamento de modelos. A divisao foi
revista uma vez para avaliar ambos os alvos. A divisao original e preservada como artefato de
auditoria em `data/interim/prior_binary_split_assignments.parquet` e
`reports/tables/prior_binary_feature_manifest.json`.

A regra para escolher a nova divisao foi definida antes da geracao final: aplicar
`StratifiedGroupKFold` sobre os cinco estratos, com cinco dobras, sementes `20261009` ate
`20261020` (12 repeticoes), e grupos `representative_repo_id`. Entre os 60 candidatos, escolher
o menor valor da soma de tres diferencas absolutas: fracao do teste em relacao a 20%, prevalencia
binaria do teste em relacao a amostra e maior diferenca entre as quatro proporcoes de intensidade
condicionais aos positivos do teste e da amostra. Empates sao resolvidos pela ordem da semente e
da dobra. A escolha usa apenas rotulos e grupos; nenhum atributo preditivo ou desempenho de
modelo e consultado. Contagens, diferencas e assinatura ficam no manifesto da divisao.

Na divisao escolhida, 160.000 conteudos ficaram no treino e 40.000 no teste; 25.918 e 6.479
sao positivos, respectivamente. Os 57.808 repositorios representativos do treino e 14.568
do teste sao disjuntos. As contagens por estrato no teste sao: 33.521 nao reutilizados,
3.648 com 2 repositorios, 1.522 com 3-4, 759 com 5-9 e 550 com 10 ou mais. A divisao
tem assinatura SHA-256
`c608febca59f3e2065ebcacd6f27e56c8b0e90230bd060b6238bd1505e15ab15`. Os detalhes
estao em `reports/tables/final_feature_manifest.json`.

O agrupamento e definido pelo repositorio *representativo* de cada conteudo, nao por todos os
repositorios onde existem copias. Portanto, a ausencia de sobreposicao desse identificador nao
elimina totalmente possiveis relacoes entre repositórios ou textos quase duplicados.

O conjunto de teste revisado nao deve ser usado para selecao de atributos, escolha de algoritmo,
ajuste de hiperparametros ou definicao do tratamento do desbalanceamento. Essas decisoes devem
ser tomadas apenas no treino, por validacao cruzada agrupada.

## Pre-processamento e experimentos iniciais (T4-T5)

Os pipelines combinam TF-IDF (unigramas e bigramas do conteudo) com atributos estruturais
numericos e metadados categoricos do repositorio. Imputacao, escalonamento, categorias e
vocabulario sao ajustados dentro de cada dobra, sem consultar a validacao daquela dobra. Alvos,
contagens que os definem, identificadores e atribuicao de split sao excluidos explicitamente dos
preditores. A representacao TF-IDF e aprendida em todo o treino da dobra e compartilhada pelos
dois modelos; o classificador de intensidade e ajustado e avaliado somente nas linhas positivas.

O primeiro modelo e regressao logistica, com comparacao pareada entre `class_weight=None` e
`class_weight="balanced"`. Para a intensidade, usa-se `saga`, adequado a matrizes esparsas, com
diagnostico explicito de convergencia. As matrizes transformadas por dobra sao persistidas e
reutilizadas entre tentativas quando fonte e parametros coincidem. A validacao usa cinco dobras
`StratifiedGroupKFold`, preservando
`representative_repo_id` como grupo e os cinco estratos da tarefa. Para reutilizacao binaria,
registram-se average precision, ROC-AUC, balanced accuracy, precisao, recall e F1 no limiar 0,5.
Para intensidade, o ajuste e condicional a conteudos reutilizados e compara F1 macro/ponderado,
balanced accuracy, acuracia e erro absoluto ordinal. As taxas de cada faixa sao registradas por
dobra.

Os pesos sao uma ablaçao experimental, nao uma suposicao de melhoria. O teste externo conserva a
distribuicao natural e nao participa da selecao do tratamento, limiar ou modelo. SMOTE e outras
amostragens sinteticas nao fazem parte do protocolo inicial; uma eventual experiencia futura
devera ocorrer somente dentro do treino de cada dobra.

## Atributos iniciais

- volume textual: caracteres, palavras, linhas, sentencas e diversidade lexical;
- estrutura Markdown: titulos, listas, blocos de codigo, links, URLs, tabelas e checkboxes;
- front matter e composicao: validade, tamanho, scripts, referencias e arquivos irmaos;
- contexto reservado para ablaçao: localizacao, linguagem, estrelas e forks do repositorio
  representativo.

O historico de commits esta ausente em aproximadamente 75,75% da amostra final. Variaveis
historicas serao usadas com indicadores explicitos de ausencia e avaliadas por ablaçao, sem
imputacao que confunda "nao coletado" com um valor observado.

## Dados finais

A amostra oficial de 13 mil conteudos e usada apenas para desenvolver e testar o pipeline. Os
resultados finais seguirao duas camadas:

1. estatisticas descritivas calculadas sobre todas as ocorrencias da tabela `artifacts`;
2. modelagem sobre uma amostra deterministica de aproximadamente 200 mil conteudos, selecionada
   somente depois do calculo populacional de reutilizacao.

A revisao do espelho Parquet e fixada em
`ebab17454a7c236f8f26b183567f1a126f42e3f8`. O teste externo da amostra de desenvolvimento nao
sera reutilizado como teste final.

### Amostragem deterministica para modelagem

O alvo e calculado antes da amostragem, contando repositorios distintos em todas as ocorrencias.
Em seguida, aplica-se MD5 a concatenacao do hash do conteudo com a semente fixa `20261009`.
Os conteudos sao ordenados por esse segundo hash e os primeiros 200 mil sao selecionados. Essa
segunda dispersao evita depender da uniformidade empirica do identificador original e fornece
uma selecao pseudoaleatoria, estavel e independente dos atributos do conteudo. A prevalencia da
classe positiva nao e fixada artificialmente: sua proximidade com a populacao e verificada e
registrada no manifesto.
