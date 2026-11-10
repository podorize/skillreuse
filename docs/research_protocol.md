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
