# 5. Discussion

The results answer both questions. The text of a skill predicts whether other developers will copy it (RQ1), and the signal comes mainly from how the skill is written (RQ2). This section discusses what that means (5.1), what else could explain it (5.2), who can use it (5.3) and where it stops (5.4).

## 5.1 What the results mean

The signal is real but modest. It is real because it holds for authors the model never saw, for skills that appeared after the ones it learned from, and for every window and definition of aggregator we tried. It is modest because the model ranks a copied skill above another one only about 65% of the time, and it never finds a skill that is very likely to be copied. It is also not a hidden measure of where a skill was published: a model that knows the origin repository, including its later popularity, still improves when it reads the text.

What the model relies on is, above all, writing style. Copied skills read like explanations: several full sentences, a stable vocabulary, longer words and a valid front matter. One natural reading is that such skills describe *general* ways of working, which transfer from one project to another, and are labeled clearly enough that an agent — or a person browsing a repository — can tell what they are for. Short, terse texts with a varied vocabulary, by contrast, often record the conventions of a single codebase and have little reason to travel. This reading is plausible, but our data cannot separate it from the alternatives below.

## 5.2 Alternative explanations

**The origin repository.** A skill in a popular repository may be copied because of where it is, not because of what it says. The control rules out the strong form of this explanation: the text adds 0.049 AUC to a model that knows the repository's age, language, license, the folder of the file, and its stars and forks. Because popularity was measured after T0, and may partly result from the copies themselves, this control favors the repository. It cannot, however, capture other kinds of visibility of the author, such as followers or activity elsewhere.

**Provenance.** Skills that travel together — collections released by a company, packs copied as a whole — may share a recognizable house style. A model that learns that style would recognize where a skill comes from, not what makes it reusable. Two results limit this explanation: the model predicts skills whose names never appear in development almost as well (Appendix A), and it predicts skills that appeared after the ones it learned from. Still, style may partly mark provenance.

**Generated text.** Long sentences and a repetitive vocabulary are also typical of text written by language models or skill generators. If such tools produce skills that are both more polished and more widely promoted, style would reflect how a skill was produced rather than the author's choices. GitSkills does not record who or what wrote a skill, so this question remains open.

## 5.3 Implications

**For researchers** using GitSkills or similar snapshots, our measurement suggests three precautions. Count only copies by other accounts, so that people copying their own skills are not mistaken for adoption. Treat repositories that collect skills in bulk separately: in GitSkills, 38% of the distinct skills exist only in such repositories, and they would otherwise dominate any count of copies. And date the copies, so that every skill has the same time to be copied. **For skill authors and curators**, the results suggest — as hypotheses to test, not as recipes — that explanatory instructions with valid metadata travel further than terse notes tied to one project. **For tool builders**, a model like ours can help rank new skills for review or recommendation, but its precision is far too low for automatic decisions.

## 5.4 Limitations

**What we measure.** A copy by another account is only an approximation of adoption. Several accounts of the same person or organization count as different developers. Installers copy skills on request; we count these copies as adoption, since someone chose to install the skill, but a skill listed in an installer's catalog has an advantage that its text does not explain. A copy is not proof of use, nor of quality. Because we compare exact contents, a skill that another developer edits becomes a new skill. Finally, the limit of 100 skills separates bulk collection from individual choice only roughly, although the result does not depend on it (Section 4.4).

**How we measure it.** We only see copies that still existed in July 2026. Copies deleted earlier are invisible, and 9.4% of the copies we tried to date had disappeared since. T0 is therefore the first appearance we can observe, which may come after the true one, and the number of copied skills is a lower bound. Histories rewritten with tools such as force-push are not detected. The popularity of the origin repository could only be measured in July 2026. The sample covers 150,000 of the 1.16 million skills, and only skills that appeared at least 60 days before the snapshot could be used, so the youngest skills are not represented.

**What it allows us to say.** The study follows the order of events, but it is observational: the models describe associations between the text and later copying, not causes. Permutation importance and SHAP describe the model, not the world, and correlated attributes share their importance. We therefore draw conclusions only from what was stable: the first place of writing style and the direction of the leading attributes.
