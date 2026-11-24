import sys
from pathlib import Path

import pytest

from reid.notebook import CommandFailed, has_finished_run, sh
from reid.storage import db, files


def test_sh_streams_output_and_returns_zero(capsys):
    assert sh([sys.executable, "-c", "print('hello')"]) is None  # no stray value in notebooks
    out = capsys.readouterr().out
    assert "hello" in out
    assert out.startswith("$ ")  # the command is echoed first


def test_sh_raises_on_failure(capsys):
    with pytest.raises(CommandFailed, match="exit code 3"):
        sh([sys.executable, "-c", "import sys; print('boom'); sys.exit(3)"])
    assert "boom" in capsys.readouterr().out  # the output that explains the failure is shown


def test_sh_shell_string():
    with pytest.raises(CommandFailed, match="exit code 2"):
        sh("exit 2")


def test_sh_accepts_paths(tmp_path: Path, capsys):
    sh([sys.executable, "-c", "import sys; print(sys.argv[1])", tmp_path])
    assert str(tmp_path) in capsys.readouterr().out


def test_has_finished_run(tmp_path: Path):
    assert not has_finished_run(tmp_path)  # no database at all
    assert not files.db_path(tmp_path).exists()  # and the check does not create one
    conn = db.connect(files.db_path(tmp_path))
    run_id = db.create_run(conn, "{}")
    assert not has_finished_run(tmp_path)
    db.set_run_status(conn, run_id, "done")
    conn.close()
    assert has_finished_run(tmp_path)
