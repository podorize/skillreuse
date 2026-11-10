# Apêndice A — Reprodutibilidade e execução

## A.1 Dados e ambiente

Fonte pública: [GitSkills no Hugging Face](https://huggingface.co/datasets/mvaccargiu/gitskills), revisão fixa `ebab17454a7c236f8f26b183567f1a126f42e3f8` (snapshot registrado como julho de 2026). O download usa `artifacts` e `repos` e verifica SHA-256. A agregação calcula `repository_count` por `file_sha` usando `COUNT(DISTINCT repo_full_name)` antes da amostragem. Os 200.000 conteúdos são os primeiros na ordenação `md5(content_hash || ':20261009'), content_hash` da população. A divisão externa agrupada por `representative_repo_id` produz 160.000 conteúdos de desenvolvimento e 40.000 de teste; entre os conteúdos reutilizados (`repository_count >= 2`), restam 25.918 e 6.479 linhas. Essa é uma amostra determinística, não uma amostra aleatória simples com erro amostral conhecido.

O ambiente usa Python 3.12, `pyproject.toml` e `uv.lock`. A amostragem usa semente `20261009`; os experimentos usam semente `2027` e três dobras `StratifiedGroupKFold`. Os comandos abaixo partem da raiz do repositório.

## A.2 Construção dos dados

```bash
uv sync
uv run skillreuse download
uv run skillreuse build
uv run skillreuse features
uv run skillreuse target
```

As entradas resultantes são `data/processed/final_modeling_sample.parquet`, `final_structured_features.parquet` e `final_targets.parquet`. Versões, cardinalidades e partições estão em `results/tables/full_population_manifest.json`, `results/tables/final_feature_manifest.json` e `results/count_regression/count_target_manifest.json`.

## A.3 Validação e avaliação

Comparação Ridge das três representações:

```bash
uv run skillreuse benchmark-views --candidate-features 10000 --top-k 500 \
  --views skill_only repository_only hybrid --models ridge_log --folds 3 --seed 2027 \
  --output-dir results/count_regression/selected_full_ridge
```

Comparação dos seis regressores sobre os atributos de repositório:

```bash
uv run skillreuse benchmark --views repository_only --folds 3 --seed 2027 \
  --output-dir results/discipline_benchmark/full_repository_only_20261109
```

O spot-check está em `results/discipline_benchmark/spot_check_repository_only_20261109/`. A avaliação externa abaixo **já foi executada**; repetir para verificar reprodutibilidade não autoriza novo ajuste pelos resultados:

```bash
uv run skillreuse evaluate \
  --output-dir results/discipline_benchmark/final_repository_only
```

As tabelas da Seção 4 derivam de `selected_full_ridge/count_regression_pilot_summary.csv`, `full_repository_only_20261109/regression_summary.csv`, `spot_check_repository_only_20261109/spot_check_summary.csv` e `final_repository_only/test_summary.csv`. `random_forest_test_predictions.csv` contém as previsões externas usadas na auditoria.

## A.4 Figuras e auditoria

```bash
uv run skillreuse report
```

O comando gera quantis e valores ausentes do desenvolvimento, referência constante calculada pela mediana do desenvolvimento, métricas por faixa das previsões externas salvas e gráficos em `results/discipline_benchmark/report_audit/`. A importância por permutação e o cenário de estrelas são recalculados **apenas no desenvolvimento** com três dobras agrupadas. `audit_manifest.json` registra fontes, sementes, cortes e limites. A auditoria separada do Ridge híbrido está em `results/count_regression/tail_audit/`.

## A.5 Verificações

```bash
uv run pytest -q tests/test_count_targets.py tests/test_preprocess.py
```

Os testes verificam a construção do alvo e a exclusão de colunas de vazamento. A reprodução integral deve conferir os manifestos de divisão, a ausência de grupos compartilhados e a correspondência entre CSVs e tabelas do relatório. Arquivos históricos de modelos fora do escopo da entrega não integram as comparações apresentadas.
