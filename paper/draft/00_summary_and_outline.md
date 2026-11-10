# Resumo e sumário do relatório

## Resumo

Este trabalho investiga a intensidade de reutilização de *Agent Skills*, definida como o número de repositórios que contêm o mesmo `SKILL.md`, entre conteúdos observados em pelo menos dois repositórios. A partir de 200.000 conteúdos do GitSkills, 25.918 skills reutilizadas compõem o desenvolvimento e 6.479 o teste externo, separados por repositório representativo. Com três dobras agrupadas, comparamos as visões *skill-only*, *repository-only* e *hybrid* com Ridge; em experimento distinto, comparamos seis regressores usando apenas atributos de repositório. O Random Forest foi escolhido pelo menor RMSLE médio na validação agrupada (0,575 ± 0,008) e alcançou 0,583 no teste, frente a 0,714 de uma referência que prediz a mediana do desenvolvimento. Contudo, seu MAE externo (3,266) superou o da referência (3,067), e todas as 324 skills acima do P95 do treino foram subestimadas. A comparação de representações sugere sinal nos atributos da skill, mas sua diferença entre visões é pequena e não possui avaliação externa. As importâncias por permutação apontam variáveis contextuais, como linguagem e histórico, com variação entre dobras; não estabelecem efeitos causais nem qualidade da skill. O estudo oferece um pipeline auditável e delimita onde a previsão e as explicações são úteis ou insuficientes.

## Sumário

1. **Introdução**
   1.1 Contexto e motivação  
   1.2 Problema de pesquisa  
   1.3 Objetivos e perguntas de pesquisa  
   1.4 Contribuições  
   1.5 Organização do trabalho
2. **Trabalhos relacionados**
   2.1 Reúso de software, clonagem e artefatos em linguagem natural  
   2.2 Agent Skills, artefatos de prompt e utilidade funcional  
   2.3 Aprendizado de máquina em engenharia de software e avaliação fora da amostra  
   2.4 Regressão de contagem, explicabilidade e lacuna da pesquisa
3. **Metodologia**
   3.1 Desenho da pesquisa  
   3.2 Dados e unidade de análise  
   3.3 Definição do alvo  
   3.4 Divisão dos dados e controle de dependência  
   3.5 Construção das variáveis  
   3.6 Representação textual e seleção de atributos  
   3.7 Modelos avaliados  
   3.8 Métricas e análise de erros  
   3.9 Explicabilidade  
   3.10 Reprodutibilidade
4. **Resultados**
   4.1 Caracterização do alvo e dos dados  
   4.2 Comparação das representações  
   4.3 Comparação das famílias de modelos  
   4.4 Análise de erros por intensidade de reutilização  
   4.5 Robustez textual  
   4.6 Explicabilidade e consistência  
   4.7 Síntese dos resultados e estado da avaliação
5. **Discussão**
   5.1 Desempenho e características dos dados  
   5.2 Comparação dos modelos e vieses indutivos  
   5.3 Explicabilidade e interpretação  
   5.4 Implicações para a engenharia de software  
   5.5 Ameaças à validade e limitações
6. **Conclusão e trabalhos futuros**

**Referências**  
**Apêndice A — Reprodutibilidade e execução**  
**Apêndice B — Uso de ferramentas de inteligência artificial**
