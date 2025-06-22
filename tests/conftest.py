from typing import Any

import pytest
from syrupy import SnapshotAssertion
from syrupy.extensions.single_file import SingleFileSnapshotExtension
from syrupy.types import SerializedData


class PythonCodeSnapshotExtension(SingleFileSnapshotExtension):
    _file_extension = "snapshot.py"

    def serialize(self, data: str, **_kwargs: Any) -> SerializedData:
        return str.encode(data)

    @classmethod
    def get_snapshot_name(cls, *, test_location: Any, index: int = 0) -> str:
        original_name = super().get_snapshot_name(test_location=test_location, index=index)
        return f"code_{original_name}"


@pytest.fixture
def snapshot_equinox_code(snapshot: SnapshotAssertion):
    return snapshot.use_extension(PythonCodeSnapshotExtension)
