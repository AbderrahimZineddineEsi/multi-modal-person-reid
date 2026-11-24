from pathlib import Path

import pytest

from reid.storage import db


@pytest.fixture
def conn(tmp_path: Path):
    c = db.connect(tmp_path / "reid.db")
    yield c
    c.close()
