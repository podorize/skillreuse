# Registro do uso de ferramentas de IA

Este arquivo documenta o uso de ferramentas de IA no projeto, conforme as diretrizes da
disciplina CMP263.

## Entrada 1 - Planejamento e preparacao do projeto

- Data: 09/10/2026
- Ferramenta: OpenAI Codex
- Finalidade: discutir o problema de pesquisa, elaborar o plano inicial e preparar a estrutura
  reproduzivel do projeto.
- Responsabilidade dos autores: revisar todas as decisoes, implementar e compreender o codigo,
  validar os resultados e produzir a analise critica e as conclusoes.
- Limitacoes observadas: as sugestoes iniciais ainda dependem da verificacao do schema real e da
  distribuicao efetiva do dataset.

## Entrada 2 - Pipeline inicial da amostra

- Data: 09/10/2026
- Ferramenta: OpenAI Codex
- Finalidade: inspecionar o schema da amostra oficial, implementar a construcao do alvo por hash
  e gerar verificacoes iniciais de qualidade e distribuicao.
- Responsabilidade dos autores: revisar as consultas, confirmar a interpretacao das colunas e
  avaliar criticamente se a amostra e adequada para os experimentos.
- Limitacoes observadas: a amostra e deterministica pelos menores hashes e sera usada inicialmente
  para desenvolvimento; resultados finais deverao explicitar se foram obtidos nela ou no dataset
  completo.

## Entrada 3 - Engenharia de atributos e divisao dos dados

- Data: 09/10/2026
- Ferramenta: OpenAI Codex
- Finalidade: implementar atributos textuais e estruturais, uma divisao estratificada por classe
  e agrupada por repositorio, testes de determinismo e relatorios de cobertura.
- Responsabilidade dos autores: revisar se cada atributo representa o conceito pretendido,
  justificar a divisao adotada e manter o teste externo intocado durante a selecao de modelos.
- Limitacoes observadas: contagens baseadas em expressoes regulares aproximam a estrutura
  Markdown e nao substituem uma analise semantica completa.

## Entrada 4 - Estrategia de dados finais

- Data: 09/10/2026
- Ferramenta: OpenAI Codex
- Finalidade: definir o uso do dataset completo para estatisticas populacionais e uma amostra
  deterministica de 200 mil conteudos para modelagem; criar download seletivo e verificavel.
- Responsabilidade dos autores: justificar o tamanho da amostra final, revisar os criterios da
  populacao e relatar a revisao exata dos dados.
- Limitacoes observadas: o armazenamento local impede manter todas as tabelas auxiliares; apenas
  `artifacts` e `repos` sao necessarias nesta fase.

## 2026-10-09 - Camada populacional

- A IA auxiliou na implementacao do download verificavel do espelho Parquet completo.
- A IA auxiliou na implementacao da agregacao global por hash e da selecao deterministica de
  200 mil conteudos para modelagem.
- A decisao metodologica foi manter a prevalencia natural e calcular o alvo antes da amostragem.
- Os resultados foram verificados por testes automatizados, invariantes e manifestos com hashes.

## 2026-10-09 - Features e divisao final

- A IA auxiliou na implementacao do processamento em lotes dos 200 mil textos.
- A IA auxiliou na identificacao e correcao de duas falhas de implementacao encontradas durante
  a execucao: coluna duplicada e variacao de tipos entre lotes.
- A IA auxiliou na construcao da divisao agrupada e estratificada de treino e teste.
- A saida foi verificada por testes automatizados e invariantes de cardinalidade, unicidade,
  ausencia de vazamento direto e isolamento dos grupos.

## 2026-10-09 - Analise exploratoria

- A IA auxiliou na adaptacao do roteiro de analise do material didatico ao conjunto GitSkills.
- A IA auxiliou na implementacao das estatisticas univariadas e multivariadas e das figuras.
- Comparacoes condicionadas ao alvo foram restringidas ao treino para preservar o teste final.
- Interpretacoes produzidas pela IA foram conferidas contra as tabelas calculadas pelo pipeline.

## 2026-10-12 - Dois alvos e revisao da divisao externa

- A IA auxiliou na formalizacao da reutilizacao binaria e da intensidade ordinal condicional.
- A IA implementou a busca deterministica de uma divisao agrupada pelos cinco estratos,
  com regra de selecao registrada antes da execucao final.
- A IA auxiliou na implementacao dos pre-processadores sem vazamento entre dobras e da
  comparacao de regressao logistica sem pesos e com pesos balanceados nos dois alvos.
- A primeira execucao integral da validacao cruzada foi interrompida antes de concluir a
  vetorizacao TF-IDF; portanto, ainda nao ha resultados de modelo para reportar.
- Os autores devem revisar os limiares da intensidade, a justificativa metodologica da
  revisao da divisao e a ausencia de vazamento de alvos para os atributos preditivos.
- A revisao do split exige recalcular posteriormente as analises que usam o subconjunto de
  treino; nao implica reutilizacao futura nem causalidade.
