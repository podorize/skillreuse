# Pipeline do relatorio. Requer `uv`; os dados vao para data/raw (~7 GB).
SEED ?= 2027
R    := results/owner
V    := skill repo+catalog clean_full clean_full+shortcuts shortcuts_only

.PHONY: html quick-setup quick interpretation feature-eda derived data owner controls primary robustness importance final holdout figures test paper

quick-setup:    ## copia as tabelas derivadas (9 MB) para data/processed, sem baixar os dados brutos
	mkdir -p data/processed
	cp -n data/derived/*.parquet data/processed/

quick: quick-setup  ## reproducao rapida (~6 min): protocolo principal (1 repeticao) e holdout em outro diretorio
	uv run skillreuse primary --repeats 1 --output-dir results/owner/primary_quick
	uv run skillreuse holdout-eval --output-dir results/owner/holdout_repro

data:           ## download (revisao fixa), amostra deterministica, divisao e alvo de contagem
	uv run skillreuse download
	uv run skillreuse build
	uv run skillreuse features
	uv run skillreuse target

owner:          ## alvo por donos, contexto, auditoria do representante e EDA
	uv run skillreuse owner

feature-eda:    ## EDA dos atributos (ausentes, assimetria, colinearidade, equilibrio)
	uv run skillreuse feature-eda

derived:        ## regenera as tabelas pequenas de data/derived a partir de data/processed
	uv run skillreuse export-derived

controls:       ## controle com o desenho original (alvo repository_count)
	uv run skillreuse controls --repeats 3 --folds 3 --seed $(SEED)

primary:        ## protocolo principal: H1-H4 com limite de 50 por dono
	uv run skillreuse primary

robustness:     ## limiares de donos, cap, classes de localizacao, alvo repository_count
	uv run skillreuse owner-models --output-dir $(R)/models
	uv run skillreuse owner-models --exclude-owners-above 500 --output-dir $(R)/models_excl_big_owners
	uv run skillreuse owner-models --group-by repo --views skill repo+catalog clean_full clean_full+shortcuts --output-dir $(R)/models_repo_grouped
	uv run skillreuse owner-models --exclude-owners-above 200 --views $(V) --output-dir $(R)/robustness/excl200
	uv run skillreuse owner-models --exclude-owners-above 1000 --views $(V) --output-dir $(R)/robustness/excl1000
	uv run skillreuse owner-models --cap-per-owner 50 --views $(V) --output-dir $(R)/robustness/cap50
	for t in skills-dir other canonical; do uv run skillreuse owner-models --tier $$t --exclude-owners-above 500 --views skill repo+catalog clean_full --output-dir $(R)/robustness/tier_$$t; done
	uv run skillreuse owner-models --cap-per-owner 50 --views clean_full minus_skill minus_repo minus_catalog minus_path --output-dir $(R)/robustness/ablation
	uv run skillreuse owner-overlap
	uv run skillreuse fork-audit
	uv run skillreuse owner-models --cap-per-owner 50 --log-skewed --views skill clean_full --output-dir $(R)/robustness/log_skewed
	uv run skillreuse owner-models --task regression --regression-target repo --exclude-owners-above 500 --views $(V) --output-dir $(R)/robustness/target_repository_count

importance:     ## importancia por permutacao (conjunto limpo)
	uv run skillreuse owner-importance --exclude-owners-above 500 --repeats 2 --n-perm 3

interpretation:  ## dependencia parcial (fora da amostra) e analise de limiar
	uv run skillreuse interpretation

final:          ## tuning aninhado e congelamento dos modelos (nao le o holdout)
	OMP_NUM_THREADS=1 uv run skillreuse final-model --repeats 2 --folds 3 --inner-folds 3 --n-iter 6 --n-jobs 4

holdout:        ## constroi o holdout disjunto por dono
	uv run skillreuse holdout

holdout-eval:   ## avaliacao UNICA no holdout (trava em results/owner/holdout/evaluation_lock.json)
	uv run skillreuse holdout-eval

figures:        ## figuras do relatorio
	uv run skillreuse figures

test:
	uv run pytest

paper:          ## Markdown -> LaTeX (requer pandoc)
	uv run python paper/build_latex.py

html:           ## versao de leitura do relatorio em HTML unico (paper/report.html)
	uv run python paper/build_html.py
