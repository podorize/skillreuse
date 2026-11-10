# Metodologia

## Pergunta de pesquisa

Entre as skills reutilizadas, em que medida caracteristicas textuais, estruturais e contextuais
permitem prever a quantidade observada de repositorios que reutilizam cada skill?

## Unidade de analise

Um conteudo distinto de `SKILL.md`, identificado por `content_hash`.

## Alvo definido

O alvo principal e `repository_count` entre skills com `reused = 1`. A modelagem considera a
contagem observada na escala original e, quando apropriado, a transformacao `log1p` para reduzir
o efeito da cauda longa. Modelos de contagem sobredispersa, como Binomial Negativa, sao candidatos
prioritarios.

`reused` continua sendo usado para selecionar a populacao condicional, mas nao e um alvo da
entrega principal. `reuse_intensity` permanece como variavel derivada para auditoria da EDA e nao
integra o protocolo principal. `repository_count` e `occurrence_count` nunca sao atributos
preditivos.

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

## Revisao da divisao externa para o alvo de contagem

A divisao original, com assinatura
`4ddc4b1b69302ef696d56c855ebc9d226bacc4599c56f5572f5bffba058bd8ae`,
equilibrava `reused`, mas a distribuicao da contagem positiva exigia auditoria adicional. Essa
limitacao foi observada antes de qualquer treinamento de modelos. A divisao foi revista uma vez
para preservar a prevalencia e a distribuicao condicional de reutilizacao. A divisao original e preservada como artefato de
auditoria em `data/interim/prior_binary_split_assignments.parquet` e
`reports/tables/prior_binary_feature_manifest.json`.

A regra para escolher a nova divisao foi definida antes da geracao final: aplicar
`StratifiedGroupKFold` sobre os cinco estratos, com cinco dobras candidatas, sementes `20261009` ate
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

## Pre-processamento e experimentos

Os pipelines combinam TF-IDF (unigramas e bigramas do conteudo) com atributos estruturais
numericos e metadados categoricos do repositorio. Imputacao, escalonamento, categorias,
vocabulario e selecao de termos sao ajustados dentro de cada dobra, sem consultar a validacao.
Alvos, contagens que os definem, identificadores e atribuicao de split sao excluidos dos
preditores.

O vocabulario textual sera gerado no treino de cada dobra com limites de 5.000, 10.000 e 20.000
candidatos e selecao de 50, 100, 200 ou 500 termos. Para a contagem continua, a triagem
univariada usa `f_regression` sobre `log1p(repository_count)`; ela tambem e refeita dentro de
cada dobra. O pool historico de 211 termos nao participa do novo pipeline. As matrizes de cada
configuracao serao versionadas para garantir reproducibilidade.

A validacao agrupada usara `representative_repo_id` e registrara RMSLE como metrica principal,
alem de MAE, erro em `log1p` e Spearman. O benchmark comparara representacoes `skill-only`,
`repository-only` e `hybrid`, seguido de tuning dos modelos mais promissores.

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
