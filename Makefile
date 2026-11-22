# Pipeline do estudo (paper/protocol.md). Requer `uv`.
# Quatro niveis de reproducao, um comando cada (README, "Como reproduzir"):
#   make verify        1. conferir os resultados        so o repositorio, minutos
#   make rerun         2. refazer a analise             so o repositorio
#   make rebuild       3. refazer o tratamento          + GitSkills (~7 GB); usa as datas publicadas
#   make from-scratch  4. refazer do zero               + GitSkills + API do GitHub (gh autenticado, horas)
.PHONY: verify rerun rebuild from-scratch quick test tables data collect paper pdf

# ---------------------------------------------------------------------------
# Niveis de reproducao
# ---------------------------------------------------------------------------

verify: quick test  ## 1. confere os modelos treinados no holdout e cada numero do artigo

rerun:          ## 2. refaz as fases 3-7 (EDA, selecao, modelos, holdout, robustez, interpretacao)
	uv run skillreuse eda
	uv run skillreuse spot
	OMP_NUM_THREADS=1 uv run skillreuse tune
	uv run skillreuse final
	uv run skillreuse freeze
	uv run skillreuse holdout --allow-reevaluation
	uv run skillreuse control
	uv run skillreuse sensitivity
	uv run skillreuse interpret
	uv run skillreuse figures

rebuild: data   ## 3. refaz a amostra e as tabelas a partir do GitSkills, com as datas publicadas, e a analise
	uv run skillreuse collect sample
	$(MAKE) tables rerun

from-scratch: data collect  ## 4. refaz a coleta das datas na API, as tabelas e a analise (resultado nao identico)
	$(MAKE) tables rerun

# ---------------------------------------------------------------------------
# Etapas
# ---------------------------------------------------------------------------

quick:          ## aplica os modelos treinados ao holdout e confere a AUC da avaliacao unica
	uv run skillreuse reproduce

test:           ## invariantes do estudo e conferencia dos numeros do artigo
	uv run pytest

tables:         ## fase 2: T0, autor, alvo, atributos e divisao; popularidade (controle); variantes (sensibilidade)
	uv run skillreuse cohort --gitskills
	uv run skillreuse cohort
	uv run skillreuse cohort --popularity
	uv run skillreuse cohort --variants

data:           ## download verificado do GitSkills (revisao fixa); pula arquivos ja conferidos
	uv run skillreuse download

collect:        ## fase 1: amostra e datas das copias (API do GitHub, horas). SOBRESCREVE as datas publicadas
	uv run skillreuse collect sample
	uv run skillreuse collect dates --scope main
	uv run skillreuse collect stars
	uv run skillreuse collect dates --scope aggregators
	uv run skillreuse collect merge

paper:          ## Markdown -> LaTeX (requer pandoc)
	uv run python paper/build_latex.py

pdf: paper      ## compila paper/latex/main.pdf (requer tectonic)
	cd paper/latex && tectonic -X compile main.tex
