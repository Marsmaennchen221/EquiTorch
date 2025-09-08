from typing import Any

import pytest
from syrupy import SnapshotAssertion
from syrupy.extensions.single_file import SingleFileSnapshotExtension


def pytest_addoption(parser):
    """Add custom command line options for pytest."""
    parser.addoption(
        "--write-transpiled",
        action="store_true",
        default=False,
        help="Write transpiled code to files in tests/transpiled/ folder",
    )


@pytest.fixture
def write_transpiled(request):
    """Fixture to access the --write-transpiled command line option."""
    return request.config.getoption("--write-transpiled")


class PythonCodeSnapshotExtension(SingleFileSnapshotExtension):
    _file_extension = "snapshot.py"

    def serialize(self, data: str, **_kwargs: Any) -> bytes:
        return str.encode(data.replace("\r\n", "\n"), encoding="utf-8")

    @classmethod
    def get_snapshot_name(cls, *, test_location: Any, index: int = 0) -> str:
        original_name = super().get_snapshot_name(test_location=test_location, index=index)
        return f"code_{original_name}"

    def matches(self, *, serialized_data: bytes, snapshot_data: bytes) -> bool:
        return serialized_data.decode(encoding="utf-8").replace("\r\n", "\n") == snapshot_data.decode(
            encoding="utf-8"
        ).replace("\r\n", "\n")


@pytest.fixture
def snapshot_equinox_code(snapshot: SnapshotAssertion):
    return snapshot.use_extension(PythonCodeSnapshotExtension)
