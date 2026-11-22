"""Statistics of the study: Holm correction, Nadeau-Bengio paired test and bootstrap by author."""

from __future__ import annotations

import numpy as np
from scipy import stats

from skillreuse.config import CONFIG

SEED = CONFIG["models"]["seed"]
N_BOOT = CONFIG["evaluation"]["n_boot"]  # bootstrap resamples of the main result


def holm(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni correction for a family of comparisons.

    The p-values are sorted from the smallest to the largest. The k-th smallest is
    multiplied by (m - k + 1), where m is the number of comparisons, and an adjusted value
    is never smaller than the one before it. The result keeps the order of the input.
    """
    m = len(p_values)
    adjusted = np.empty(m)
    largest_so_far = 0.0
    for rank, index in enumerate(np.argsort(p_values)):
        largest_so_far = max(largest_so_far, (m - rank) * p_values[index])
        adjusted[index] = min(1.0, largest_so_far)
    return adjusted.tolist()


def nb_test(diff: np.ndarray, ratio: float) -> dict[str, float]:
    """Paired t test with the Nadeau-Bengio correction, on the differences between two models.

    ``diff`` has one difference of AUC per validation fold. The training sets of the folds
    overlap, so the differences are not independent and the usual variance is too small.
    The correction multiplies the variance by (1/n + n_validation/n_training), where
    ``ratio`` is n_validation/n_training. The test has n - 1 degrees of freedom.
    """
    n = len(diff)
    mean = float(np.mean(diff))
    variance = float(np.var(diff, ddof=1))
    std_error = np.sqrt((1 / n + ratio) * variance)
    t_critical = stats.t.ppf(0.975, n - 1)
    if std_error > 0:
        p_value = float(2 * stats.t.sf(abs(mean / std_error), n - 1))
    else:
        p_value = 0.0 if mean else 1.0
    return {"mean_diff": mean, "ci95_low": mean - t_critical * std_error,
            "ci95_high": mean + t_critical * std_error, "p_nb": p_value}


def owner_bootstrap(owners: np.ndarray, statistic, n_boot: int = N_BOOT, seed: int = SEED
                    ) -> np.ndarray:
    """Bootstrap that resamples whole authors, not single skills.

    Skills of the same author are not independent, so in each resample the authors are
    drawn with replacement, and every skill gets as weight the number of times its author
    was drawn (0 leaves the author out). ``statistic`` receives these weights, one per row,
    and returns a number, for example the AUC with ``sample_weight``.
    """
    authors, author_of_row = np.unique(owners, return_inverse=True)
    rng = np.random.default_rng(seed)
    results = []
    for _ in range(n_boot):
        drawn = rng.integers(0, len(authors), len(authors))
        times_drawn = np.bincount(drawn, minlength=len(authors))
        weights = times_drawn[author_of_row].astype(float)
        results.append(statistic(weights))
    return np.asarray(results)


def boot_p(samples: np.ndarray, null: float) -> float:
    """Two-sided bootstrap p-value: twice the share of resamples on the side of ``null``,
    with a +1 correction so that it is never exactly 0 (with 2,000 resamples the smallest
    possible value is about 0.001)."""
    share_below = (np.sum(samples <= null) + 1) / (len(samples) + 1)
    share_above = (np.sum(samples >= null) + 1) / (len(samples) + 1)
    return float(min(1.0, 2 * min(share_below, share_above)))
