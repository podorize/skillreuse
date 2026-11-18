"""Testes da correcao de Holm."""

import numpy as np

from skillreuse.stats import holm


def test_holm_is_monotone_and_never_below_raw_p() -> None:
    p = [0.001, 0.04, 0.03, 0.2]
    adjusted = holm(p)
    assert all(a >= q for a, q in zip(adjusted, p, strict=True))
    order = np.argsort(p)
    assert all(adjusted[order[i]] <= adjusted[order[i + 1]] for i in range(len(p) - 1))
    assert adjusted[0] == 0.004

