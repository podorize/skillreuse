# Apêndice A — Reprodutibilidade e execução

## A.1 Dados, ambiente e sementes

Fonte pública: [GitSkills no Hugging Face](https://huggingface.co/datasets/mvaccargiu/gitskills), revisão fixa `ebab17454a7c236f8f26b183567f1a126f42e3f8` (julho de 2026); o download verifica SHA-256. O ambiente usa Python 3.12, `pyproject.toml` e `uv.lock`. Sementes: amostragem `20261009`; validação 2027, 2028 e 2029; holdout `20261110`; bootstrap 2027. A data de referência para idade e recência é 2026-07-20. A amostra de 200.000 conteúdos é determinística (`md5(content_hash || ':20261009')`), não uma amostra aleatória simples. Os comandos partem da raiz do repositório; `make` encapsula cada etapa.

## A.2 Construção dos dados e auditoria

```bash
uv sync
make data      # download, amostra, divisão e alvo de contagem
make owner     # owner_count, atributos de contexto, auditoria do representante, EDA
```

Saídas: `data/processed/owner_targets.parquet` (contagens por conteúdo na população completa), `context_features.parquet` e, em `results/owner/`, `eda_summary.json`, `eda_target_quantiles.csv`, `eda_owner_concentration.csv`, `audit_by_repository_count.csv`, `audit_by_owner_count.csv`. O manifesto da EDA registra a concordância da regra do representante (97,2%).

## A.3 Protocolo principal e robustez

```bash
make controls     # controle com o desenho original (Seção 4.2)
make primary      # H1–H4 (Seção 4.3), limite de 50 por dono
make robustness   # limiares de donos, classes de localização, alvo alternativo (4.4)
make importance   # importância por permutação (4.5)
```

`results/owner/primary/` contém `primary_hypotheses.csv`, as tabelas por dobra e os testes pareados exploratórios; `results/owner/robustness/` e `results/owner/models*/` guardam cada sensibilidade, cada uma com `manifest.json`.

## A.4 Tuning, congelamento e holdout

```bash
make final         # tuning aninhado e congelamento (Seção 4.6); não lê o holdout
make holdout       # constrói o holdout disjunto por dono (Seção 3.9)
make holdout-eval  # avaliação única (Seção 4.7)
```

Os modelos congelados ficam em `results/owner/final/final_config_*.json` e foram commitados antes da avaliação (tag git `frozen-before-holdout`). `holdout-eval` grava `results/owner/holdout/evaluation_lock.json` e **recusa** nova execução sem `--allow-reevaluation`; qualquer reavaliação deve ser declarada. A reprodução deve conferir `holdout_manifest.json` (sobreposição zero de donos e de conteúdos).

## A.5 Mapa entre tabelas, figuras e artefatos

| Item | Artefato |
|---|---|
| Distribuição e concentração (4.1) | `results/owner/eda_target_quantiles.csv`, `eda_owner_concentration.csv` |
| Viés do representante (4.2, figura) | `results/owner/audit_by_repository_count.csv`, `eda_owner_and_selection.png` |
| Controle com o desenho original (4.2) | `results/controls/shortcut_summary.csv`, `shortcut_paired.csv` |
| Visões e hipóteses (4.3) | `results/owner/primary/{classification,regression}_summary.csv`, `primary_hypotheses.csv` |
| Sensibilidade (4.4) | `results/owner/figures/sensitivity_owner_limit.csv`, `results/owner/robustness/*` |
| Importância (4.5) | `results/owner/importance/*_importance.csv` |
| Tuning (4.6) | `results/owner/final/*_nested_summary.csv`, `*_selection_tests.csv` |
| Holdout (4.7) | `results/owner/holdout/holdout_results.json` |

: Origem dos números do relatório.

## A.6 Verificações, desvios e limites

```bash
make test
```

Os testes cobrem a exclusão de atributos contaminados, os padrões de caminho, os testes pareados e a correção de Holm, o bootstrap por dono e a disjunção do holdout. Desvios em relação ao protocolo: tuning com 2×3 dobras e 6 candidatas (planejado 3×3 e 12) por custo computacional; a máquina de desenvolvimento tinha 8 GB de memória. Resultados numéricos podem variar nas últimas casas com versões diferentes das bibliotecas. A primeira versão do estudo (alvo `repository_count`, teste externo original) permanece no repositório (`results/count_regression`, `results/discipline_benchmark`) apenas como histórico, e não sustenta nenhuma conclusão deste relatório.
