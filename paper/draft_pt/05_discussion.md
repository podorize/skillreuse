# 5. Discussão

## 5.1 O que o sinal significa

Os atributos observáveis de uma skill e de seu repositório carregam informação sobre o reúso entre donos, mas pouca: AUC 0,73 no holdout e uma redução de 6% do RMSLE frente à média de `log1p`. O resultado se sustenta nas três camadas de evidência (validação aninhada, robustez do desenvolvimento e holdout disjunto por dono), o que o distingue de uma associação de uma única partição. Ele é, porém, um sinal de **ordenação grossa**, e não de previsão fina: o Spearman do modelo de regressão é 0,33.

A decomposição por blocos responde à RQ3. Skill (estrutura do arquivo) e repositório (visibilidade e atividade) contribuem de forma complementar: isolados, cada um alcança AUC 0,65–0,67; juntos, 0,72 (H3a e H3b). A importância por permutação associa o reúso entre donos sobretudo à **visibilidade do repositório** (estrelas, forks, recência do último push), o que é coerente com skills em repositórios mais vistos serem copiadas por outros donos. A associação admite mais de uma explicação (visibilidade que gera cópias, ou repositórios valorizados por conterem skills já copiadas), e os dados não as separam. O peso da licença e do tamanho do catálogo sugere um componente de ecossistema (catálogos e clones) que merece investigação própria.

## 5.2 Implicações para o uso do GitSkills

A contribuição mais sólida deste trabalho é metodológica e responde à RQ1.

1. **`location_class` e `history_fetched` não são preditores legítimos de reúso.** Por construção, a classe de localização do representante cresce com o número de cópias (14% → 92%), e o histórico de commits herda a dependência. Estudos que os usem para explicar a difusão correm o risco de reportar um artefato; na primeira versão deste estudo, eles faziam a visão "de repositório" parecer muito mais informativa do que é.
2. **Contar repositórios mistura difusão com autocópia.** O número de donos distintos é um alvo mais fiel, e a escolha altera a base da distribuição (28% de conteúdos com um único dono).
3. **Resultados dependem de poucas contas.** O AUC variou de 0,615 a 0,747 conforme o tratamento de donos entre 500 e 1000 conteúdos. Relatar a sensibilidade a donos concentrados, ou limitar a contribuição por dono, deveria ser prática padrão.
4. **O representante descreve uma cópia arbitrária.** Entre cópias da mesma classe, a escolha é alfabética, e os atributos do repositório (estrelas, linguagem) são os de um repositório entre vários. Agregados sobre todas as cópias seriam mais informativos, mas derivam das mesmas ocorrências que definem o alvo; recomendamos que sejam tratados explicitamente como vazamento em estudos preditivos, ou calculados só com informação anterior ao reúso.

## 5.3 Modelos e limites dos dados

As seis famílias empatam em ambas as tarefas, e a busca de hiperparâmetros não melhorou o resultado (RF padrão 0,716 contra 0,701 buscado). Quando famílias de vieses indutivos tão diferentes (linear, vizinhança, árvores, boostings) chegam ao mesmo desempenho, o limite costuma estar nos dados e não no algoritmo. Os atributos descrevem uma cópia do conteúdo, e o alvo é uma contagem exata de hashes, que ignora cópias modificadas; ganhos relevantes provavelmente exigirão novas fontes de informação (relações entre cópias, tempo), e não modelos mais flexíveis.

## 5.4 Ameaças à validade e limitações

**Construto.** O número de donos aproxima a adoção independente, mas contas podem representar a mesma pessoa ou organização, e a busca do GitHub só conta forks com mais estrelas que o pai. As duas contagens são limites inferiores, e o hash exato não captura variantes. Reúso por cópia não indica uso, qualidade ou utilidade da skill.

**Validade interna.** (i) As hipóteses H1–H4 foram fixadas depois de resultados exploratórios, e só o holdout é confirmatório; os valores do desenvolvimento são descritivos. (ii) O limite de 50 conteúdos por dono é uma escolha dependente dos dados. (iii) O tuning usou 2 repetições de 3 dobras e 6 candidatas, menos que o planejado. (iv) Com 9 escores não independentes, o teste de Nadeau–Bengio é aproximado, e os p-valores após Holm dependem do tamanho da família. (v) Os blocos de catálogo e caminho vêm do repositório e do caminho do representante e podem ser parcialmente afetados pela preferência por locais padrão; o fato de o caminho quase não ajudar e de o sinal persistir dentro de cada classe de localização reduz esse risco, mas não o elimina.

**Validade externa.** O corpus é um instantâneo de julho de 2026 de repositórios públicos. O holdout exclui donos concentrados por construção, tem mais skills `canonical` (36% contra 21%) e menos `other` (9% contra 21%) que o desenvolvimento, e H3 e H4 não foram reavaliadas nele. A partição de teste original foi consultada na primeira versão do estudo e não sustenta nenhuma afirmação confirmatória.

**Interpretação.** A importância por permutação é associativa, depende do modelo e é instável entre dobras para vários atributos. Nada aqui estabelece causalidade.

**Ética.** Os dados são públicos e o GitSkills anonimiza autores de commits; omitimos identificadores de donos, e a análise por dono não visa avaliar indivíduos.
