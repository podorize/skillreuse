# Fase 4 — explicabilidade inicial e auditoria de estabilidade

## Execução

- Modelo: Ridge sobre `log1p(repository_count)`.
- Visão: `hybrid`.
- Vocabulário: 10.000 candidatos; 500 termos selecionados por dobra.
- População: 25.918 skills reutilizadas e 7.701 grupos da partição externa de treino.
- Teste externo: não utilizado.

Artefatos principais:

- `explainability_hybrid/global_feature_importance.csv`;
- `explainability_hybrid/local_explanations.csv`;
- `explainability_hybrid/feature_stability.csv`;
- `explainability_hybrid/fold_feature_coefficients.csv`.

## Achados

Os coeficientes globais e as contribuições locais foram calculados na escala de
`log1p(repository_count)`. A estabilidade em três dobras foi medida por recorrência do termo e
consistência do sinal.

Os sinais mais estáveis incluem termos como `date_added`, `2026`, `02`, `https` e fragmentos de
documentação. Isso é um alerta metodológico: o modelo está capturando padrões temporais, URLs ou
convenções editoriais que podem refletir o processo de coleta ou a origem dos documentos, e não
propriedades semânticas intrínsecas da skill.

A auditoria lexical classificou 500 termos textuais selecionados: 7 (1,4%) possuem padrão
numérico/data e nenhum foi classificado como URL ou identificador longo. A fração é pequena, mas
esses termos ocupam posições muito altas na importância global; portanto, a decisão não pode ser
baseada apenas na proporção de tokens.

## Parecer crítico

Ainda não é válido afirmar que os termos mais importantes “explicam a reutilização” como
características substantivas da skill. A estabilidade entre dobras demonstra recorrência
preditiva, mas não elimina confusão temporal, artefatos de documentação ou dependência do corpus.

Antes do teste externo, a pesquisa precisa executar uma auditoria de robustez textual:

1. quantificar tokens numéricos, datas, URLs e identificadores;
2. comparar o modelo atual com uma versão que mascara esses padrões;
3. verificar se as conclusões estruturais/contextuais permanecem estáveis;
4. manter os termos textuais apenas se o ganho e a interpretação sobreviverem à comparação.

O primeiro artefato de auditoria está em `text_artifact_audit.csv` e seu manifesto em
`text_artifact_audit_manifest.json`.

## Robustez com texto mascarado

Foi repetida a avaliação completa removendo datas, números, URLs e identificadores longos antes
do TF-IDF. Os resultados foram:

| Visão | Texto | RMSLE | MAE |
|---|---|---:|---:|
| hybrid | bruto | 0,568 | 3,32 |
| hybrid | artefatos removidos | 0,570 | 3,33 |
| skill-only | bruto | 0,572 | 3,33 |
| skill-only | artefatos removidos | 0,575 | 3,34 |

A diferença é pequena, indicando que o desempenho preditivo não depende principalmente desses
artefatos. Na explicabilidade mascarada, nenhum dos 500 termos selecionados foi classificado como
data, número, URL ou identificador longo. Ainda assim, termos editoriais como `date_added` e
`source community` permanecem e devem ser interpretados como padrões de documentação, não como
efeitos causais.

O próximo item da auditoria é estratificar o erro por quantis do alvo, sobretudo no segmento de
alta reutilização, antes da avaliação externa.

As explicações locais devem ser usadas para auditoria de casos e não como evidência causal.
