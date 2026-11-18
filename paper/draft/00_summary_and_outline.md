# Abstract and contents

## Abstract

Agent skills are Markdown files that teach LLM-based coding agents how to perform a task, and they spread across GitHub by being copied. We ask whether the text of a skill predicts whether other developers will copy it. To measure copying reliably, we dated the copies of 150,000 GitSkills skills through the GitHub API and labeled a skill as copied when another developer copied it within 60 days of its first appearance, ignoring repositories that collect skills in bulk. On 20,564 skills from unseen authors, a model of the text reached an AUC of 0.645, beyond text length and beyond the popularity of the origin repository. Skills written as explanatory prose, with a stable vocabulary and valid metadata, were copied more often. The signal is modest but robust, also on future skills.

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
   5.2 Alternative explanations  
   5.3 Implications  
   5.4 Limitations
6. **Conclusion**

**References**  
**Appendix A — Reproducibility**  
**Appendix B — Use of artificial intelligence tools**
