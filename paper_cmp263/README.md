# Artigo SBC — CMP263 (UFRGS)

Esta pasta contém o artigo SBC e os materiais suplementares preparados para a disciplina de **Aprendizagem de Máquina (CMP263 - INF/UFRGS)**. `main.tex` e `references.bib` são as fontes a compilar e submeter. `REPORT_CMP263.md` é um rascunho de relatório separado e não integra o pacote Overleaf.

## Formatos Disponíveis:

1. **Artigo em LaTeX (Padrão SBC - Sociedade Brasileira de Computação)**:
   * Arquivo principal: [`main.tex`](main.tex)
   * Arquivo de referências: [`references.bib`](references.bib)
   * Estilo e bibliografia SBC: [`sbc-template.sty`](sbc-template.sty), [`sbc.bst`](sbc.bst)
   * Figuras: pasta [`figures/`](figures/)
   * Suplemento de uso de IA: [`AI_USAGE_SUPPLEMENT.md`](AI_USAGE_SUPPLEMENT.md)
   * Pacote Overleaf gerado: `../sbc_paper_overleaf.zip`

2. **Reprodução dos resultados**:
   * [Notebook de Replicação no Colab](https://colab.research.google.com/github/podorize/skillreuse/blob/main/notebooks/reproduce_in_colab.ipynb)
   * Localmente, execute `python -m gitskills_reuse.reproduce` com as dependências instaladas e caches disponíveis; a última avaliação do cache levou aproximadamente 499 segundos, incluindo 20 reamostragens bootstrap. A geração das previsões OOF para o pool fixo de 211 termos levou aproximadamente 1.174 segundos. Tempos variam conforme o hardware.
   * Para reconstruir o cache versionado de 211 termos, use `python -m gitskills_reuse.reproduce --from-scratch`. A opção `--from-scratch --align-cv-text-pool` reconstrói o vocabulário de até 50.000 termos usado nas dobras de desenvolvimento e pode demandar bastante memória e tempo; os resultados dessa variante precisam ser avaliados e atualizados antes de substituir os números deste artigo.
