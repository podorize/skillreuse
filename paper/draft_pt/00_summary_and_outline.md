# Resumo e sumário do relatório

## Resumo

Agent Skills são copiadas entre repositórios sem registro central, e o GitSkills documenta essa difusão em larga escala. Investigamos se características observáveis de uma skill e de seu repositório se associam ao reúso entre donos distintos no GitHub e quanto dessa previsibilidade é artefato da construção do dataset. A regra que escolhe o representante de cada conteúdo faz a classe de localização depender do número de cópias: a chance de o representante estar em `.claude/skills/` sobe de 14% (2 repositórios) para 92% (mais de 50), enquanto a fração real de cópias nessa pasta fica entre 5% e 13%. Além disso, 28% das skills reutilizadas existem dentro de um único dono, e poucas contas concentram a amostra. Com atributos livres desses artefatos, validação agrupada por dono, tuning aninhado e um holdout novo disjunto por dono (25.011 skills de 10.484 donos), o Random Forest atinge AUC 0,73 (IC95% 0,72–0,75) para reúso entre donos, e um HistGradientBoosting reduz o RMSLE do número de donos em 6% frente à média (0,625 contra 0,664). O sinal é modesto, robusto a variações do protocolo e vem de skill e repositório de forma complementar; é associativo, não causal.

## Sumário

1. **Introdução**
   1.1 Contexto e motivação  
   1.2 Problema de pesquisa  
   1.3 Objetivos e perguntas de pesquisa  
   1.4 Contribuições  
   1.5 Organização do trabalho
2. **Trabalhos relacionados**
   2.1 Reúso, clonagem e artefatos em linguagem natural  
   2.2 Agent Skills e artefatos de prompt  
   2.3 Validação, vazamento e viés de construção  
   2.4 Modelagem, explicabilidade e lacuna da pesquisa
3. **Metodologia**
   3.1 Desenho da pesquisa  
   3.2 Dados e unidade de análise  
   3.3 Regra do representante e auditoria  
   3.4 Alvos  
   3.5 Atributos  
   3.6 Validação e limite por dono  
   3.7 Modelos, baselines e tuning  
   3.8 Métricas e inferência  
   3.9 Holdout novo  
   3.10 Interpretação
4. **Resultados**
   4.1 O alvo e a concentração por donos  
   4.2 O viés de seleção do representante  
   4.3 Protocolo principal  
   4.4 Robustez  
   4.5 Interpretação  
   4.6 Seleção de modelo e tuning  
   4.7 Confirmação no holdout
5. **Discussão**
   5.1 O que o sinal significa  
   5.2 Implicações para o uso do GitSkills  
   5.3 Modelos e limites dos dados  
   5.4 Ameaças à validade e limitações
6. **Conclusão e trabalhos futuros**

**Referências**  
**Apêndice A — Reprodutibilidade e execução**  
**Apêndice B — Uso de ferramentas de inteligência artificial**
