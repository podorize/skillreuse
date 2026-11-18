# Pipeline do estudo (paper/protocol.md). Requer `uv`; os dados brutos vao para data/raw (~7 GB).
# A fase 1 usa a API do GitHub (gh autenticado); as tabelas da coleta estao em data/derived/collection.
.PHONY: quick data collect study holdout robustness test paper pdf

quick:          ## reproducao rapida sem dados brutos nem API: confere os modelos congelados no holdout
	uv run skillreuse reproduce

data:           ## download verificado do GitSkills (revisao fixa)
	uv run skillreuse download

collect:        ## fase 1: amostra, datas das copias e estrelas em T0 (API do GitHub, horas)
	uv run skillreuse collect sample
	uv run skillreuse collect dates --scope main
	uv run skillreuse collect stars
	uv run skillreuse collect dates --scope aggregators
	uv run skillreuse collect merge

study:          ## fases 2-5: autores e divisao, EDA e janela, selecao, modelos finais e congelamento
	uv run skillreuse cohort
	uv run skillreuse eda
	uv run skillreuse spot
	OMP_NUM_THREADS=1 uv run skillreuse tune
	uv run skillreuse final
	uv run skillreuse freeze

holdout:        ## fase 6: avaliacao UNICA (trava em results/phase6/evaluation_lock.json) e controle
	uv run skillreuse holdout
	uv run skillreuse cohort --popularity
	uv run skillreuse control

robustness:     ## fase 7: sensibilidades, interpretacao e figuras
	uv run skillreuse cohort --variants
	uv run skillreuse sensitivity
	uv run skillreuse interpret
	uv run skillreuse figures

test:
	uv run pytest

paper:          ## Markdown -> LaTeX (requer pandoc)
	uv run python paper/build_latex.py

pdf: paper      ## compila paper/latex/main.pdf (requer tectonic)
	cd paper/latex && tectonic -X compile main.tex
