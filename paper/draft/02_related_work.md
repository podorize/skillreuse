# 2. Related work

This study draws on three lines of work: software reuse, agent skills, and good practices for machine learning on mined data.

In software reuse research, reuse has long been studied through component adoption and the spread of solutions across open-source projects \cite{mockus2002two,haefliger2008code}.
Large-scale GitHub analyses showed that duplicated files are everywhere \cite{lopes2017dejavu}, and that code is reused through dependencies, copy-and-paste and templates \cite{heinemann2011extents}.
Like these studies, we identify copies through exact file-content comparison, but unlike them, we focus on natural-language artifacts and track each copy over time and across accounts to distinguish adoption by other developers from self-copying and bulk collection.

Research on language models using tools has focused on how a model select and invokes those tools \cite{schick2023toolformer,shen2023hugginggpt,yao2023react,qin2023toolllm}.
Agent skills, in contrast, are persistent, human-readable files stored in repositories \cite{agentskills_spec}.
GitSkills offers the first large collection of them \cite{gitskills2027}.
Recent studies examine barriers to skill reuse \cite{zhang2026keepskillsreuse}, test whether skills help agents solve real tasks \cite{han2026sweskillsbench}, and track prompt evolution within software projects \cite{tafreshipour2025prompting}. To our knowledge, however, no study has examined whether skill text predicts adoption by other developers.

In machine learning on mined data, related examples, such as files from the same account, appear in both the training and the test data, a model can memorize the group instead of learning a general relation \cite{kaufman2012leakage}.
Group-based splits are the standard remedy \cite{saeb2017need,roberts2017cross}.
Models can also exploit accidental patterns that happen to correlate with the target \cite{geirhos2020shortcut,lapuschkin2019unmasking}, which in mined datasets can arise from collection procedures.
Comparing models across repeated cross-validation needs a correction for the overlap between training sets \cite{nadeau2003inference} and, when many comparisons are made, for multiple testing \cite{holm1979simple}.
To interpret a model, permutation importance measures how much its performance depends on each attribute \cite{breiman2001statistical,molnar2020interpretable}, and SHAP values split each prediction into attributes contributions \cite{lundberg2017unified,lundberg2020local}.
Both explain the model, not the world \cite{shmueli2010explain}, and we use them in that sense.
