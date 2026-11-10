# 1. Introdução

## 1.1 Contexto e motivação

A adoção de agentes baseados em Grandes Modelos de Linguagem (LLMs) na engenharia de software vem sendo acompanhada pelo surgimento de novos artefatos destinados a encapsular conhecimento operacional. Entre eles estão as *Agent Skills*: diretórios portáveis que combinam metadados estruturados, instruções em linguagem natural e, eventualmente, scripts e referências. O arquivo `SKILL.md` é o núcleo descritivo do artefato e orienta o agente sobre como executar uma tarefa, quais restrições observar e quais recursos usar \cite{agentskills_spec}. As skills ampliam a noção tradicional de reúso, pois distribuem procedimentos e conhecimento de domínio sem depender de uma biblioteca com interface formal.

O reúso de software é estudado há décadas \cite{haefliger2008code,lopes2017dejavu}, mas a difusão de Agent Skills tem características próprias: não há registro central nem gerenciador de pacotes, e o reúso ocorre por cópia de pastas entre repositórios. Uma mesma skill pode aparecer em muitos repositórios porque é clara e portável, porque pertence a um template ou catálogo, porque foi instalada por uma ferramenta, ou porque uma mesma conta a duplicou em vários de seus projetos. Essas explicações misturam propriedades do artefato e do contexto de distribuição, e dificultam interpretar a reutilização observada.

O GitSkills disponibiliza um corpus em larga escala com ocorrências de skills, conteúdos únicos e metadados de repositórios \cite{gitskills2027}. Outros estudos examinam defeitos de especificação \cite{zhang2026keepskillsreuse}, o efeito de skills em tarefas de engenharia de software \cite{han2026sweskillsbench} e a evolução de prompts \cite{tafreshipour2025prompting}. Resta uma questão complementar e metodologicamente delicada: quais características observáveis estão associadas à difusão de uma skill entre donos distintos, e o quanto essa previsibilidade resulta da forma como o próprio dataset foi construído?

## 1.2 Problema de pesquisa

Operacionalizamos o reúso pela presença de conteúdo idêntico de `SKILL.md` (mesmo `content_hash`) em repositórios distintos. Uma primeira versão deste estudo modelava o número de repositórios (`repository_count`) a partir de atributos da skill, de texto e do repositório representativo, com validação agrupada por repositório e um teste externo. Duas auditorias, descritas neste relatório, mostraram que esse desenho era frágil.

A primeira diz respeito ao **representante**. O GitSkills guarda metadados detalhados de apenas uma cópia de cada conteúdo, escolhida pela melhor classe de localização (preferindo `.claude/skills/`) e, depois, pela ordem alfabética do repositório e do caminho. Como um conteúdo com mais cópias tem mais chances de ter alguma cópia na pasta preferida, a classe de localização do representante, e o indicador de histórico de commits que depende dela, passam a codificar o próprio alvo. A segunda diz respeito à **autocópia e à concentração**: 28% das skills reutilizadas têm todas as cópias dentro de um único dono, e uma conta responde sozinha por 12% do conjunto de desenvolvimento. Contar repositórios mistura difusão entre desenvolvedores com duplicação por uma mesma conta.

Reformulamos, então, o problema. O alvo principal passa a ser o **reúso entre donos** (`cross_owner`: o conteúdo aparece sob dois ou mais donos distintos) e, secundariamente, o número de donos (`owner_count`). Os atributos passam a excluir as variáveis que dependem da regra do representante, que são analisadas como um achado. A avaliação usa validação agrupada por dono, um limite de conteúdos por dono, tuning aninhado e hipóteses pré-especificadas, e a confirmação é feita em um **holdout novo, disjunto por dono**, avaliado uma única vez. Como a pesquisa passou por revisão após resultados exploratórios, distinguimos explicitamente o que é confirmatório do que é exploratório. As conclusões são associações observacionais \cite{shmueli2010explain}: não afirmamos causalidade nem qualidade da skill.

## 1.3 Objetivos e perguntas de pesquisa

O objetivo geral é investigar, de forma reprodutível, quais características observáveis se associam ao reúso de Agent Skills entre donos e o quanto essa associação é afetada por artefatos de construção do dataset.

* **RQ1 — Vieses de construção:** em que medida a regra de escolha do representante, a autocópia e a concentração por donos confundem o alvo e os atributos do GitSkills?
* **RQ2 — Previsibilidade:** com atributos livres desses artefatos, é possível prever o reúso entre donos e o número de donos melhor que baselines, e a estimativa se confirma em um holdout disjunto por dono?
* **RQ3 — Origem do sinal:** qual é a contribuição de skill, repositório e catálogo, quais variáveis se associam às previsões e com que consistência?

## 1.4 Contribuições

1. Uma **auditoria do representante** do GitSkills: recuperamos sua regra (97,2% de concordância) e mostramos que a localização e o histórico dependem do número de cópias.
2. Um **protocolo** para modelar o reúso entre donos, com alvo por donos, limite de conteúdos por dono, validação agrupada, hipóteses pré-especificadas com correção de Holm e confirmação em holdout novo.
3. Evidência de um **sinal preditivo modesto e robusto**, decomposto em blocos (skill, repositório, catálogo, caminho) e confirmado em dados nunca vistos: AUC 0,73 para reúso entre donos.
4. Um **pacote reprodutível** que registra revisão dos dados, sementes, partições e a trava que impede reutilizar o holdout.

## 1.5 Organização do trabalho

A Seção 2 apresenta os trabalhos relacionados; a 3, a metodologia; a 4, os resultados; a 5, a discussão e as ameaças à validade; a 6, as conclusões. Os apêndices tratam da reprodução e do uso de IA.
