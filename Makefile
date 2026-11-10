# Pipeline completo do relatorio. Requer `uv` e os dados em data/raw (ver `make data`).
SEED ?= 2027
RES  := results

.PHONY: data views benchmark evaluate report test paper clean-cache

data:           ## download (revisao fixa), amostra deterministica, divisao e alvo de contagem
	uv run skillreuse download
	uv run skillreuse build
	uv run skillreuse features
	uv run skillreuse target

views:          ## Ridge: skill-only, repository-only e hybrid
	uv run skillreuse benchmark-views --candidate-features 10000 --top-k 500 \
	  --views skill_only repository_only hybrid --models ridge_log --folds 3 --seed $(SEED) \
	  --output-dir $(RES)/count_regression/selected_full_ridge

benchmark:      ## 6 regressores, validacao agrupada (repository-only)
	uv run skillreuse benchmark --views repository_only --folds 3 --seed $(SEED) \
	  --output-dir $(RES)/discipline_benchmark/full_repository_only_20261109

evaluate:       ## avaliacao externa dos candidatos congelados (ja executada uma vez)
	uv run skillreuse evaluate --output-dir $(RES)/discipline_benchmark/final_repository_only

report:         ## tabelas e figuras do relatorio
	uv run skillreuse report

test:
	uv run pytest

paper:          ## Markdown -> LaTeX (requer pandoc)
	uv run python paper/build_latex.py
