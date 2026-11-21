"""Statistics of the study: Holm correction, Nadeau-Bengio paired test and author bootstrap."""

from __future__ import annotations

import numpy as np
from scipy import stats


def holm(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni correction for a family of comparisons.

    The p-values are sorted from the smallest to the largest. The k-th smallest is
    multiplied by (m - k + 1), where m is the number of comparisons, and the adjusted
    values are forced to never decrease. This controls the chance of a false positive in
    the whole family, and is never more conservative than the Bonferroni correction.
    The result keeps the order of the input.
    """
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values))
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


SEED = 2027
N_BOOT = 2000  # bootstrap resamples of the main result


def nb_test(diff: np.ndarray, ratio: float) -> dict[str, float]:
    """Paired t test with the Nadeau-Bengio correction, on the differences between two models.

    ``diff`` has one difference of AUC per validation fold. The training sets of the folds
    overlap, so the differences are not independent and the usual variance is too small.
    The correction multiplies the variance by (1/n + n_validation/n_training), where
    ``ratio`` is n_validation/n_training. The test has n - 1 degrees of freedom.
    Returns the mean difference, its 95% confidence interval and the p-value.
    """
    n, m = len(diff), float(np.mean(diff))
    var = float(np.var(diff, ddof=1))
    se = np.sqrt((1 / n + ratio) * var)
    crit = stats.t.ppf(0.975, n - 1)
    p = float(2 * stats.t.sf(abs(m / se), n - 1)) if se > 0 else (0.0 if m else 1.0)
    return {"mean_diff": m, "ci95_low": m - crit * se, "ci95_high": m + crit * se, "p_nb": p}


def owner_bootstrap(owners: np.ndarray, fn, n_boot: int = N_BOOT, seed: int = SEED
                    ) -> np.ndarray:
    """Bootstrap that resamples whole authors, not single skills.

    Skills of the same author are not independent, so resampling skills one by one would
    give intervals that are too narrow. In each resample the authors are drawn with
    replacement; the number of times an author was drawn becomes the weight of each of
    their skills (a weight of 0 removes the author). ``fn`` receives the weights of all
    rows and returns the statistic, for example the AUC with ``sample_weight``.
    Returns the statistic of each resample.
    """
    codes, inv = np.unique(owners, return_inverse=True)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, len(codes), len(codes)), minlength=len(codes))[inv]
        out.append(fn(w.astype(float)))
    return np.asarray(out)


def boot_p(samples: np.ndarray, null: float) -> float:
    """Two-sided bootstrap p-value: twice the share of resamples on the side of ``null``
    opposite to the effect, with a +1 correction so that p is never exactly 0.

    With 2,000 resamples and none on the wrong side, p = 2 * 1/2001, about 0.001.
    """
    below = (np.sum(samples <= null) + 1) / (len(samples) + 1)
    above = (np.sum(samples >= null) + 1) / (len(samples) + 1)
    return float(min(1.0, 2 * min(below, above)))
