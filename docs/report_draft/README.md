# Rascunho do relatório

Os arquivos desta pasta são a fonte de desenvolvimento textual do relatório da disciplina. O conteúdo é mantido em Markdown para facilitar revisão e discussão antes da migração ao template do Overleaf.

## Arquivos

- `PLAN_REVISION.md`: plano de correções prioritárias para fechar o relatório da disciplina;
- `00_summary_and_outline.md`: resumo revisado e sumário;
- `01_introduction.md`: Introdução;
- `02_related_work.md`: Trabalhos Relacionados;
- `03_methodology.md`: Metodologia;
- `04_results.md`: Resultados, incluindo o benchmark completo da disciplina;
- `05_discussion.md`: Discussão, incluindo ameaças à validade e limitações;
- `06_conclusion.md`: Conclusão e trabalhos futuros;
- `08_appendix_reproducibility.md`: Apêndice de reprodutibilidade;
- `09_appendix_ai_usage.md`: Apêndice sobre uso de IA.

## Estado

As seções foram revisadas para o trabalho da disciplina. Resultados, Discussão e Conclusão incorporam o benchmark, os dois resultados externos `repository-only`, uma referência simples, a auditoria de cauda e a importância por permutação. As figuras e CSVs correspondentes estão em `reports/modeling/discipline_benchmark/report_audit/`. O arquivo `06_validity.md` é apenas nota auxiliar; a versão do relatório está na Seção 5.5. A formatação final no template da disciplina e a revisão visual do PDF ainda não foram realizadas.

## Migração ao Overleaf

Na etapa de formatação, os títulos de primeiro nível devem ser convertidos em `\\section{}` e os títulos de segundo nível em `\\subsection{}`. As 22 chaves de citação usadas no rascunho foram conferidas contra `paper_cmp263/references.bib`; nenhuma está ausente. A lista de referências deve ser gerada no template. As figuras deste rascunho devem ser copiadas da pasta de auditoria para o projeto LaTeX, com legendas e fontes. Conferir coluna única, limite de 20 páginas incluindo referências e declaração e suplemento de uso de IA.
