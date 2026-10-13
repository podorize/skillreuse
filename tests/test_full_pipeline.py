import hashlib

import duckdb

from gitskills_reuse.data.full import select_deterministic_hashes


def test_deterministic_hash_selection_is_stable_and_natural(tmp_path) -> None:
    population = tmp_path / "population.parquet"
    connection = duckdb.connect()
    connection.execute(
        """
        COPY (
            SELECT * FROM (VALUES
                ('d', 1),
                ('a', 0),
                ('c', 1),
                ('b', 0)
            ) AS contents(content_hash, reused)
        ) TO ? (FORMAT PARQUET)
        """,
        [str(population)],
    )

    first = select_deterministic_hashes(connection, population, 3)
    second = select_deterministic_hashes(connection, population, 3)

    assert first.equals(second)
    expected = sorted(
        ["a", "b", "c", "d"],
        key=lambda value: (hashlib.md5(f"{value}:20261009".encode()).hexdigest(), value),
    )[:3]
    assert first["content_hash"].tolist() == expected
