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
    assert "[done in " in out  # and its duration printed last


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


def test_notebook_kernel_never_imports_numpy():
    """Kaggle/Colab installs upgrade numpy on disk while the kernel already holds the old one, so
    the notebook kernel may only import the standard library, IPython and ``reid.notebook``
    (which itself must not pull in numpy); everything else runs as ``reid`` commands."""
    import ast
    import json
    import subprocess

    code = "import sys, reid.notebook; sys.exit('numpy' in sys.modules or 'pandas' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
    allowed = {"reid.notebook", "IPython.display"}
    nb = Path(__file__).parents[1] / "notebooks" / "kaggle_aicity_benchmark.ipynb"
    for cell in json.loads(nb.read_text(encoding="utf-8"))["cells"]:
        src = "".join(cell["source"])
        if cell["cell_type"] != "code" or src.lstrip().startswith("%%"):
            continue
        for node in ast.walk(ast.parse(src)):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            for name in names:
                top = name.split(".")[0]
                assert name in allowed or top in sys.stdlib_module_names, name
