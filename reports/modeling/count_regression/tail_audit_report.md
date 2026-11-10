# Auditoria da cauda de reutilização

## Configuração

- Ridge em `log1p(repository_count)`;
- visão `hybrid`;
- 10.000 candidatos TF-IDF e 500 termos;
- três dobras `StratifiedGroupKFold` por `representative_repo_id`;
- limites P50/P75/P90/P95 calculados somente no treino de cada dobra.

## Resultados internos

| Faixa do alvo | Linhas | Mediana real | Mediana prevista | RMSLE | MAE | Subestimação |
|---|---:|---:|---:|---:|---:|---:|
| <=P50 | 14.593 | 2 | 3,47 | 0,432 | 1,46 | 6,7% |
| P50–P75 | 6.087 | 3 | 3,28 | 0,302 | 1,11 | 48,3% |
| P75–P90 | 2.773 | 6 | 3,73 | 0,466 | 2,25 | 89,7% |
| P90–P95 | 1.232 | 11 | 4,51 | 0,890 | 7,20 | 99,7% |
| >P95 | 1.233 | 33 | 6,22 | 1,692 | 34,75 | 100,0% |

## Parecer

O RMSLE médio mascara uma falha importante: o modelo reproduz razoavelmente a região central,
mas subestima sistematicamente a cauda de alta reutilização. Para `>P95`, a previsão mediana é
6,2 contra 33 observados e todas as instâncias são subestimadas.

Consequentemente, o Ridge não deve ser congelado ainda como modelo final para a pergunta sobre
intensidade de reutilização. A próxima etapa deve comparar estratégias que tratem explicitamente
a cauda (pesos definidos no treino, calibração multiplicativa ou modelos de contagem
sobredispersa) e medir o compromisso entre RMSLE global e erro nos quantis altos.

O teste externo permanece intocado.
