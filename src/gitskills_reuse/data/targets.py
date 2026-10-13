"""Define os dois alvos e seleciona a divisao externa da amostra final."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

SOURCE_COLUMNS = ["content_hash", "representative_repo_id", "repository_count", "reused"]
INTENSITY_LABELS = {
    0: "2 repositorios",
    1: "3-4 repositorios",
    2: "5-9 repositorios",
    3: "10+ repositorios",
}
STRATA = ("not_reused", "reuse_2", "reuse_3_4", "reuse_5_9", "reuse_10_plus")


def build_targets(source: pd.DataFrame) -> pd.DataFrame:
    """Cria intensidade condicional, sem usar contagem de ocorrencias como substituto."""
    missing = set(SOURCE_COLUMNS) - set(source.columns)
    if missing:
        raise ValueError(f"Colunas ausentes para os alvos: {sorted(missing)}")
    if source["content_hash"].isna().any() or source["content_hash"].duplicated().any():
        raise ValueError("content_hash deve ser unico e nao nulo")
    if source["representative_repo_id"].isna().any():
        raise ValueError("representative_repo_id nao pode ser nulo")

    counts = pd.to_numeric(source["repository_count"], errors="coerce")
    if counts.isna().any() or (counts < 1).any() or (counts != counts.round()).any():
        raise ValueError("repository_count deve ser inteiro positivo")
    reused = pd.to_numeric(source["reused"], errors="coerce")
    if reused.isna().any() or not reused.isin([0, 1]).all():
        raise ValueError("reused deve ser binario e nao nulo")
    if not reused.eq(counts.ge(2).astype(int)).all():
        raise ValueError("reused nao corresponde a repository_count >= 2")

    targets = source[SOURCE_COLUMNS].copy()
    targets["repository_count"] = counts.astype("int32")
    targets["reused"] = reused.astype("int8")
    intensity = pd.Series(pd.NA, index=targets.index, dtype="Int8")
    intensity.loc[counts.eq(2)] = 0
    intensity.loc[counts.between(3, 4)] = 1
    intensity.loc[counts.between(5, 9)] = 2
    intensity.loc[counts.ge(10)] = 3
    targets["reuse_intensity"] = intensity

    stratum = pd.Series("not_reused", index=targets.index, dtype="string")
    for code, name in enumerate(STRATA[1:]):
        stratum.loc[intensity.eq(code).fillna(False)] = name
    targets["split_stratum"] = stratum
    return targets


def _candidate_score(
    targets: pd.DataFrame,
    test_indices: np.ndarray,
    test_size: float,
    overall_positive_rate: float,
    overall_conditional_rates: np.ndarray,
) -> tuple[float, float, float, float]:
    test = targets.iloc[test_indices]
    size_gap = abs(len(test) / len(targets) - test_size)
    binary_gap = abs(float(test["reused"].mean()) - overall_positive_rate)
    positive_intensity = test.loc[test["reused"].eq(1), "reuse_intensity"]
    if positive_intensity.empty:
        raise ValueError("Candidato sem exemplos reutilizados")
    conditional_rates = (
        positive_intensity.value_counts(normalize=True)
        .reindex(range(4), fill_value=0)
        .to_numpy(dtype=float)
    )
    conditional_gap = float(np.max(np.abs(conditional_rates - overall_conditional_rates)))
    return (size_gap + binary_gap + conditional_gap, size_gap, binary_gap, conditional_gap)


def make_two_target_split(
    targets: pd.DataFrame,
    test_size: float = 0.20,
    random_seed: int = 20261009,
    restarts: int = 12,
) -> tuple[pd.Series, dict[str, object]]:
    """Seleciona candidato agrupado por regra fixa, sem consultar modelos ou resultados."""
    if not 0 < test_size < 0.5:
        raise ValueError("test_size deve estar entre 0 e 0.5")
    if restarts < 1:
        raise ValueError("restarts deve ser positivo")
    required = set(SOURCE_COLUMNS + ["reuse_intensity", "split_stratum"])
    if required - set(targets.columns):
        raise ValueError(f"Alvos incompletos: {sorted(required - set(targets.columns))}")
    if targets["content_hash"].duplicated().any():
        raise ValueError("content_hash duplicado nos alvos")
    if targets["representative_repo_id"].isna().any():
        raise ValueError("Grupos ausentes nos alvos")
    if set(targets["split_stratum"].dropna().unique()) != set(STRATA):
        raise ValueError("Todos os cinco estratos devem estar presentes")

    ordered = targets.sort_values("content_hash").reset_index(drop=True)
    positive_rate = float(ordered["reused"].mean())
    conditional_rates = (
        ordered.loc[ordered["reused"].eq(1), "reuse_intensity"]
        .value_counts(normalize=True)
        .reindex(range(4), fill_value=0)
        .to_numpy(dtype=float)
    )
    n_splits = max(2, round(1 / test_size))
    best: tuple[tuple[float, float, float, float, int, int], np.ndarray] | None = None
    for restart in range(restarts):
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=random_seed + restart
        )
        for fold, (_, test_indices) in enumerate(
            splitter.split(
                ordered,
                y=ordered["split_stratum"],
                groups=ordered["representative_repo_id"],
            )
        ):
            score = _candidate_score(
                ordered, test_indices, test_size, positive_rate, conditional_rates
            )
            key = (*score, restart, fold)
            if best is None or key < best[0]:
                best = (key, test_indices)
    if best is None:
        raise RuntimeError("Nenhuma divisao candidata foi gerada")

    key, test_indices = best
    ordered_assignments = pd.Series("train", index=ordered.index, dtype="string")
    ordered_assignments.iloc[test_indices] = "test"
    split_by_hash = pd.Series(ordered_assignments.to_numpy(), index=ordered["content_hash"])
    assignments = targets["content_hash"].map(split_by_hash).astype("string")
    assignments.index = targets.index

    train_groups = set(targets.loc[assignments.eq("train"), "representative_repo_id"])
    test_groups = set(targets.loc[assignments.eq("test"), "representative_repo_id"])
    overlap = train_groups & test_groups
    if overlap:
        raise RuntimeError(f"Ha {len(overlap)} grupos compartilhados")
    summary = intensity_split_summary(targets.assign(split=assignments))
    if (summary["count"] <= 0).any():
        raise RuntimeError("Uma classe de intensidade ficou sem exemplos em treino ou teste")
    counts = {
        split: {row["split_stratum"]: int(row["count"]) for _, row in part.iterrows()}
        for split, part in summary.groupby("split")
    }
    signature = hashlib.sha256(
        targets.sort_values("content_hash")["content_hash"]
        .to_csv(index=False, header=False, lineterminator="\n")
        .encode("utf-8")
    ).hexdigest()
    manifest: dict[str, object] = {
        "strategy": "StratifiedGroupKFold over five strata; fixed-score candidate selection",
        "random_seed": random_seed,
        "restarts": restarts,
        "candidate_count": restarts * n_splits,
        "selected_restart": key[4],
        "selected_fold": key[5],
        "selection_score": round(key[0], 8),
        "score_components": {
            "absolute_test_fraction_gap": round(key[1], 8),
            "absolute_binary_prevalence_gap": round(key[2], 8),
            "maximum_conditional_intensity_gap": round(key[3], 8),
        },
        "requested_test_fraction": test_size,
        "actual_test_fraction": round(float(assignments.eq("test").mean()), 8),
        "train_rows": int(assignments.eq("train").sum()),
        "test_rows": int(assignments.eq("test").sum()),
        "overall_positive_rate": round(positive_rate, 8),
        "train_positive_rate": round(
            float(targets.loc[assignments.eq("train"), "reused"].mean()), 8
        ),
        "test_positive_rate": round(float(targets.loc[assignments.eq("test"), "reused"].mean()), 8),
        "train_groups": len(train_groups),
        "test_groups": len(test_groups),
        "group_overlap": 0,
        "stratum_counts": counts,
        "sample_hash_signature_sha256": signature,
    }
    return assignments, manifest


def intensity_split_summary(assignments: pd.DataFrame) -> pd.DataFrame:
    """Frequencias dos cinco estratos e taxas condicionais dos quatro positivos."""
    rows = []
    for split in ("train", "test"):
        part = assignments.loc[assignments["split"].eq(split)]
        positive_count = int(part["reused"].sum())
        counts = part["split_stratum"].value_counts()
        for stratum in STRATA:
            count = int(counts.get(stratum, 0))
            rows.append(
                {
                    "split": split,
                    "split_stratum": stratum,
                    "count": count,
                    "percentage_of_split": count / len(part) if len(part) else 0.0,
                    "percentage_among_reused": (
                        count / positive_count
                        if stratum != "not_reused" and positive_count
                        else pd.NA
                    ),
                }
            )
    return pd.DataFrame(rows)
