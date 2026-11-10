# Abstract and contents

## Abstract

Agent skills are copied between repositories without a central registry, and the GitSkills dataset documents this diffusion at scale. We investigate whether observable characteristics of a skill and of its repository are associated with reuse across distinct GitHub owners, and how much of this predictability is an artifact of how the dataset was built. The rule that selects the representative copy of each content makes its location class depend on the number of copies: the probability that the representative sits in `.claude/skills/` rises from 14% (2 repositories) to 92% (more than 50), while the actual share of copies in that folder stays between 5% and 13%. In addition, about a quarter of the reused skills (24.7% in the full population) exist under a single owner, and a few accounts concentrate the sample. With attributes free of these artifacts, owner-grouped validation, nested tuning and a new owner-disjoint holdout (25,011 skills from 10,484 owners), a Random Forest reaches an AUC of 0.73 (95% CI 0.72–0.75) for cross-owner reuse, and a HistGradientBoosting model reduces the RMSLE of the number of owners by 6% relative to the mean baseline (0.625 versus 0.664). The signal is modest and associative, not causal: the RMSLE gain is stable across protocol variations, whereas the AUC is sensitive to how concentrated owners are treated; skill and repository attributes contribute complementarily (an exploratory finding).

## Contents

1. **Introduction**
   1.1 Context and motivation  
   1.2 Research problem  
   1.3 Objectives and research questions  
   1.4 Contributions
2. **Related work**
   2.1 Reuse, cloning and natural-language artifacts  
   2.2 Agent skills and prompt artifacts  
   2.3 Validation, leakage and construction bias  
   2.4 Modeling, explainability and research gap
3. **Methodology**
   3.1 Research design  
   3.2 Data and unit of analysis  
   3.3 Representative rule and audit  
   3.4 Targets  
   3.5 Features  
   3.6 Validation and per-owner cap  
   3.7 Models, baselines and tuning  
   3.8 Metrics, inference and interpretation  
   3.9 New holdout
4. **Results**
   4.1 The target, the features and owner concentration  
   4.2 Selection bias of the representative  
   4.3 Main protocol  
   4.4 Robustness  
   4.5 Interpretation  
   4.6 Model selection and tuning  
   4.7 Confirmation on the holdout
5. **Discussion**
   5.1 What the signal means  
   5.2 Implications for using GitSkills  
   5.3 Models and data limits  
   5.4 Threats to validity and limitations
6. **Conclusion and future work**

**References**  
**Appendix A — Reproducibility and execution**  
**Appendix B — Use of artificial intelligence tools**
