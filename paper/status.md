# Estado do projeto (10/11/2026)

## Pesquisa

| Etapa | Estado | Evidência |
|---|---|---|
| Auditoria do dataset (regra do representante, autocópia, concentração) | concluída | `results/owner/audit_*`, `eda_*` |
| Alvo por donos e atributos de contexto | concluído | `data/processed/owner_targets.parquet`, `context_features.parquet` |
| Protocolo principal (H1–H4, limite de 50 por dono) | concluído | `paper/protocol.md`, `results/owner/primary/` |
| Robustez, importância e controle do desenho original | concluídos | `results/owner/robustness/`, `importance/`, `results/controls/` |
| Tuning aninhado e congelamento | concluído (tag `frozen-before-holdout`) | `results/owner/final/` |
| Holdout novo e avaliação única | concluídos | `results/owner/holdout/` |

## Relatório (`paper/draft`, em inglês; versão em português em `paper/draft_pt`)

- Seções 1–6 e apêndices A–B reescritos com a nova narrativa (viés de construção + sinal modesto confirmado).
- **Pendente:** compilar o LaTeX no Overleaf e conferir o limite de 20 páginas (a estimativa por contagem de palavras é de cerca de 20 páginas incluindo referências); revisão visual; conferência final das referências; preparar a apresentação (01/12) e as duas revisões por pares (24–30/11).

## Prazos (CMP263)

Código e artigo: 22/11/2026, 23:59. Revisão por pares: 24–30/11. Apresentação: 01/12.
