from typing import Any

import pytest
from syrupy import SnapshotAssertion
from syrupy.extensions.single_file import SingleFileSnapshotExtension
from syrupy.types import SerializedData


class PythonCodeSnapshotExtension(SingleFileSnapshotExtension):
    _file_extension = "py"

    def serialize(self, data: str, **_kwargs: Any) -> SerializedData:
        return str.encode(data)

@pytest.fixture
def snapshot_equinox_code(snapshot: SnapshotAssertion):
    return snapshot.use_extension(PythonCodeSnapshotExtension)
