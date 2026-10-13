import pytest

from gitskills_reuse.data.download_full import RemoteFile, parse_remote_files


def test_parse_remote_files_filters_and_orders_parquets() -> None:
    payload = [
        {
            "type": "file",
            "path": "data/artifacts/part-00001.parquet",
            "size": 20,
            "lfs": {"oid": "b" * 64},
        },
        {
            "type": "file",
            "path": "data/artifacts/part-00000.parquet",
            "size": 10,
            "lfs": {"oid": "a" * 64},
        },
        {"type": "file", "path": "data/repos/part-00000.parquet", "size": 1},
    ]

    files = parse_remote_files(payload, "artifacts")

    assert files == [
        RemoteFile("data/artifacts/part-00000.parquet", 10, "a" * 64),
        RemoteFile("data/artifacts/part-00001.parquet", 20, "b" * 64),
    ]


def test_parse_remote_files_rejects_missing_checksum() -> None:
    payload = [
        {
            "type": "file",
            "path": "data/artifacts/part-00000.parquet",
            "size": 10,
            "lfs": {},
        }
    ]

    with pytest.raises(ValueError, match="Metadados incompletos"):
        parse_remote_files(payload, "artifacts")
