# Supplementary Material: Generative AI Use

This supplement accompanies the CMP263 paper. OpenAI Codex was used in the early project phases for research planning, code and analysis assistance, and manuscript review; the tool is cited in the paper's references. In the later revision phase (owner-level target, grouped validation, nested tuning, new holdout and the rewrite of the report), Claude Code (Anthropic) was used as an implementation and analysis assistant.

## Representative prompts retained in the project history

The following prompts are verbatim examples from the paper-review portion of the project. They are a documented subset, not a complete prompt transcript:

1. “First, I want you to get to know the Skill Reuse research. Please, investigate the folder till you get a comprehensive understanding of the research.”
2. “Now, based on this comprehensive understanding, I want you to evaluate the new Introduction section from sbc paper. Be concise.”
3. “Fix the citations. Then, let's move to the Related Work section.”
4. “Evaluate the Related Work Section critically. Correct what is missing and send the new version, if correction is needed.”
5. “Agora implemente o plano de revisao do relatório” (revisão dos arquivos Markdown, auditoria dos resultados e documentação de reprodutibilidade).

Additional verbatim prompts from the later revision phase (Claude Code session, 2026-11-10):

6. “Certo! Agora, quero que você avalie a pesquisa criticamente com base no Enunciado em anexo.”
7. “Quero aplicar os três testes baratos sugeridos. Após finalizado, discutimos os resultados.”
8. “Reavaliar resultados”
9. “Definir protocolo principal”
10. “Siga pelo holdout e depois pelo tuning.”
11. “Reescrever relatório”

The project log at `paper/ai_usage_log.md` records additional AI-assisted work on data construction, feature processing, grouped splits, exploratory analysis, and manuscript preparation. The original prompt transcripts for many earlier implementation sessions were not retained in the repository; their task descriptions in the log are summaries, not verbatim prompts. The fifth prompt above is from the current report-revision session.

## Human review and verification

The authors are responsible for the research questions, operational definitions, data and split decisions, model configuration, interpretation, and conclusions. AI-suggested code and prose require review. In particular, the authors should verify code against the actual data schema, rerun the saved analysis pipeline, reconcile all numerical results with generated artifacts, and validate every bibliographic entry against its source. Model explanations and literature summaries are aids to review, not independent evidence.

## Known limitations of AI assistance

- AI-generated or AI-edited citations can be inaccurate or mismatched to the cited claim; references were checked and corrected during manuscript review.
- Generated code can make incorrect assumptions about column names, missing values, grouping, or data splits. The project log records implementation issues that required correction.
- Generated interpretations can overstate observational associations as causal or quality claims. The manuscript was revised to distinguish exact-content replication from adoption, utility, and skill quality.
- This supplement does not reproduce every prompt, response, model version, or intermediate code suggestion from earlier project sessions. It should not be read as a complete interaction log.
