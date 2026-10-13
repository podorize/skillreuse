import sqlite3

from gitskills_reuse.data.sample import build_analytical_table, build_quality_report


def create_fixture(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE artifacts (
            repo_full_name TEXT,
            path TEXT,
            filename TEXT,
            location_class TEXT,
            file_sha TEXT,
            content TEXT,
            frontmatter_valid INTEGER,
            name TEXT,
            description TEXT,
            body_chars INTEGER,
            history_fetched INTEGER,
            composition_fetched INTEGER,
            dedup_primary INTEGER,
            first_commit_at TEXT,
            last_commit_at TEXT,
            commit_count INTEGER,
            sibling_count INTEGER,
            sibling_bytes INTEGER,
            has_scripts INTEGER,
            has_references INTEGER,
            composition_truncated INTEGER
        );
        CREATE TABLE repos (
            full_name TEXT,
            owner TEXT,
            stars INTEGER,
            forks INTEGER,
            is_fork INTEGER,
            language TEXT,
            license TEXT,
            created_at TEXT,
            pushed_at TEXT,
            metadata_fetched INTEGER
        );
        """
    )
    rows = [
        ("repo/a", "skills/x/SKILL.md", "SKILL.md", "skills-dir", "hash-a", "A", 1,
         "a", "a", 1, 0, 1, 1, None, None, None, 0, 0, 0, 0, 0),
        ("repo/a", "copy/x/SKILL.md", "SKILL.md", "other", "hash-a", None, None,
         None, None, None, 0, 0, 0, None, None, None, None, None, None, None, None),
        ("repo/b", "skills/x/SKILL.md", "SKILL.md", "skills-dir", "hash-a", None, None,
         None, None, None, 0, 0, 0, None, None, None, None, None, None, None, None),
        ("repo/c", "skills/y/SKILL.md", "SKILL.md", "skills-dir", "hash-b", "B", 1,
         "b", "b", 1, 0, 1, 1, None, None, None, 0, 0, 0, 0, 0),
    ]
    connection.executemany(
        "INSERT INTO artifacts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        rows,
    )
    connection.executemany(
        "INSERT INTO repos VALUES (?,?,?,?,?,?,?,?,?,?)",
        [
            ("repo/a", "repo", 10, 0, 0, "Python", "MIT", None, None, 1),
            ("repo/b", "repo", 5, 0, 0, "Python", "MIT", None, None, 1),
            ("repo/c", "repo", 1, 0, 0, "Rust", "MIT", None, None, 1),
        ],
    )


def test_reuse_counts_distinct_repositories_not_occurrences() -> None:
    connection = sqlite3.connect(":memory:")
    create_fixture(connection)

    table = build_analytical_table(connection, reuse_threshold=2).set_index("content_hash")

    assert table.loc["hash-a", "occurrence_count"] == 3
    assert table.loc["hash-a", "repository_count"] == 2
    assert table.loc["hash-a", "reused"] == 1
    assert table.loc["hash-b", "repository_count"] == 1
    assert table.loc["hash-b", "reused"] == 0


def test_quality_report_detects_duplicate_hash_within_repository() -> None:
    connection = sqlite3.connect(":memory:")
    create_fixture(connection)
    table = build_analytical_table(connection, reuse_threshold=2)

    report = build_quality_report(connection, table).set_index("metric")["value"]

    assert report["distinct_hashes"] == 2
    assert report["duplicate_hash_repository_groups"] == 1
    assert report["hashes_without_exactly_one_representative"] == 0
    assert report["reused_contents"] == 1
