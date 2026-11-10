# Nota auxiliar — Ameaças à validade e limitações

Esta nota não integra a sequência numerada do relatório. A versão corrente das ameaças à validade está na Seção 5.5 de `05_discussion.md`.

## 6.1 Validade de construção

A variável `repository_count` mede a presença de conteúdo idêntico em repositórios distintos. Ela não mede intenção de adoção, uso efetivo, qualidade, utilidade, sucesso de execução ou transferência de conhecimento. Cópias modificadas e equivalentes semânticos não são capturados. Por isso, os resultados devem ser descritos como associações com replicação exata, e não como explicações completas da reutilização de software.

## 6.2 Validade interna

O agrupamento por `representative_repo_id` elimina a sobreposição desse identificador entre treino e validação, mas não garante que todos os repositórios relacionados, organizações ou cadeias de cópia estejam separados. Além disso, metadados do repositório podem funcionar como proxies de popularidade, origem ou mecanismos de distribuição. O protocolo reduz o vazamento direto, mas não elimina todas as dependências possíveis.

As etapas ajustáveis são realizadas dentro das dobras, incluindo TF-IDF e seleção de termos. Decisões metodológicas sucessivas podem ainda introduzir escolhas baseadas em resultados internos. O teste externo já foi usado uma vez para avaliar os dois candidatos `repository-only` congelados e não pode servir a novas escolhas.

Após o congelamento dos candidatos, a partição externa foi usada para avaliar os modelos finais da visão `repository-only`. A avaliação não deve ser interpretada como uma validação independente em novos snapshots; essa generalização requer replicações futuras em dados coletados em outros períodos ou ecossistemas.

## 6.3 Validade estatística

O alvo é assimétrico e sobredisperso, e a maior parte das observações está concentrada em valores baixos. O RMSLE reduz a influência dos extremos, mas pode ocultar erros absolutos elevados. Por isso, reportamos também MAE, Spearman e auditorias por quantis. A subestimação observada acima do P95 mostra que nenhuma métrica isolada é suficiente para caracterizar o desempenho.

O benchmark principal usa as 25.918 observações reutilizadas do desenvolvimento. A análise de permutação tem apenas uma repetição por atributo e dobra; diferenças pequenas de importância não devem ser tratadas como conclusões estáveis.

## 6.4 Validade externa

Os dados provêm de um snapshot específico do GitHub e de um ecossistema particular de Agent Skills. A generalização para outros períodos, hosts, comunidades, formatos de skill ou mecanismos de distribuição não é garantida. A divisão por repositório representativo avalia uma forma de generalização para grupos não observados, mas não substitui uma avaliação temporal ou uma replicação independente.

## 6.5 Validade das explicações

Importâncias por impureza e permutação dependem do modelo, da representação e do conjunto de referência. Variáveis correlacionadas podem compartilhar importância, e termos textuais podem representar templates ou origens comuns. As explicações devem ser tratadas como hipóteses observacionais, não como efeitos causais ou regras de autoria.
