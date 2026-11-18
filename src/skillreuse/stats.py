"""Estatistica do estudo: Holm, teste pareado de Nadeau-Bengio e bootstrap por dono."""

from __future__ import annotations

import numpy as np
from scipy import stats


def holm(p_values: list[float]) -> list[float]:
    """Correcao de Holm-Bonferroni para uma familia de comparacoes."""
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values))
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


SEED = 2027
N_BOOT = 2000


def nb_test(diff: np.ndarray, ratio: float) -> dict[str, float]:
    """Diferenca pareada com correcao de Nadeau-Bengio (variancia * (1/n + n_val/n_treino))."""
    n, m = len(diff), float(np.mean(diff))
    var = float(np.var(diff, ddof=1))
    se = np.sqrt((1 / n + ratio) * var)
    crit = stats.t.ppf(0.975, n - 1)
    p = float(2 * stats.t.sf(abs(m / se), n - 1)) if se > 0 else (0.0 if m else 1.0)
    return {"mean_diff": m, "ci95_low": m - crit * se, "ci95_high": m + crit * se, "p_nb": p}


def owner_bootstrap(owners: np.ndarray, fn, n_boot: int = N_BOOT, seed: int = SEED
                    ) -> np.ndarray:
    codes, inv = np.unique(owners, return_inverse=True)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, len(codes), len(codes)), minlength=len(codes))[inv]
        out.append(fn(w.astype(float)))
    return np.asarray(out)


def boot_p(samples: np.ndarray, null: float) -> float:
    """p bilateral do bootstrap: 2 x fracao do lado oposto ao efeito (com correcao +1)."""
    below = (np.sum(samples <= null) + 1) / (len(samples) + 1)
    above = (np.sum(samples >= null) + 1) / (len(samples) + 1)
    return float(min(1.0, 2 * min(below, above)))
