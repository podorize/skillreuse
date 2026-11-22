# Abstract and contents

## Abstract

Agent skills are Markdown files that teach coding agents how to perform a task. We ask whether the text of a skill predicts whether other developers will copy it. To measure copying, we dated the copies of 150,000 GitSkills skills through the GitHub API and labeled a skill as copied if another developer copied it within 60 days of its first appearance, ignoring repositories that collect skills in bulk. On 20,564 skills from unseen authors, a logistic regression on 22 text attributes reached an AUC of 0.645 (95% CI 0.624–0.666), and the result held across time windows, other definitions of bulk repositories and future skills. Writing style carried most of the signal: explanatory prose, a repetitive vocabulary and valid metadata went with more copying. The relation is modest and associative, not causal.

\

**Resumo.**

Skills de agentes são arquivos Markdown que ensinam agentes de programação a executar uma tarefa. Investigamos se o texto de uma skill prevê se outros desenvolvedores vão copiá-la. Para medir as cópias, datamos, pela API do GitHub, as cópias de 150.000 skills do GitSkills e consideramos uma skill copiada se outro desenvolvedor a copiou em até 60 dias após sua primeira aparição, ignorando repositórios que coletam skills em massa. Em 20.564 skills de autores não vistos, uma regressão logística com 22 atributos do texto alcançou AUC de 0,645 (IC 95% 0,624–0,666), e o resultado se manteve com outras janelas de tempo, outras definições de repositórios de coleta em massa e skills futuras. O estilo de escrita concentrou a maior parte do sinal: prosa explicativa, vocabulário repetitivo e metadados válidos acompanharam mais cópias. A relação é modesta e associativa, não causal.

## Contents

1. **Introduction**
2. **Related work**
3. **Materials and methods**
   3.1 Data  
   3.2 Measuring whether a skill was copied  
   3.3 Attributes  
   3.4 Training and test data  
   3.5 Exploratory analysis and preprocessing  
   3.6 Model development  
   3.7 Evaluation  
   3.8 Interpretation
4. **Results**
   4.1 Exploratory analysis  
   4.2 Choosing a model  
   4.3 Does the text predict whether a skill is copied? (RQ1)  
   4.4 How robust is the result?  
   4.5 What the model learned (RQ2)
5. **Discussion**
   5.1 What the results mean  
   5.2 Implications  
   5.3 Limitations
6. **Conclusion**

**References**  
**Appendix A — Reproducibility**  
**Appendix B — Use of artificial intelligence tools**
