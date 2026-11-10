# Pipeline completo do relatorio. Requer `uv` e os dados em data/raw (ver `make data`).
SEED ?= 2027
RES  := results

.PHONY: data owner primary views tune benchmark evaluate report test paper clean-cache

data:           ## download (revisao fixa), amostra deterministica, divisao e alvo de contagem
	uv run skillreuse download
	uv run skillreuse build
	uv run skillreuse features
	uv run skillreuse target

owner:          ## alvo por donos, contexto, auditoria do representante e EDA
	uv run skillreuse owner

primary:        ## protocolo principal: H1-H4 com limite de 50 por dono (paper/protocol.md)
	uv run skillreuse primary

views:          ## Ridge nas 3 visoes + referencias constantes, 3x3 dobras, testes pareados
	uv run skillreuse benchmark-views --candidate-features 10000 --top-k 500 \
	  --views skill_only repository_only hybrid --models dummy_median_log dummy_log ridge_log \
	  --folds 3 --repeats 3 --seed $(SEED) --output-dir $(RES)/count_regression/views_repeated

tune:           ## busca aninhada (3x3 externas, 12 candidatas) + referencias + testes pareados
	uv run skillreuse tune --view repository_only --repeats 3 --folds 3 --inner-folds 3 \
	  --n-iter 12 --seed $(SEED)

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
