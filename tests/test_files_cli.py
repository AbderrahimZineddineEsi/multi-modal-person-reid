import pytest

from reid import cli
from reid.storage import db, files


def test_paths_and_reset(tmp_path):
    root = tmp_path / "d"
    files.ensure_dirs(root)
    p = files.crop_path(root, 1234)
    assert p == root / "crops" / "00001" / "t00001234.jpg"
    assert files.rel_to_data(root, p) == "crops/00001/t00001234.jpg"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"x")
    db.connect(files.db_path(root)).close()
    assert files.data_stats(root)["crops"] == 1

    with pytest.raises(ValueError):
        files.reset_data_dir(root, confirm=False)
    assert p.exists()
    files.reset_data_dir(root, confirm=True)
    assert not p.exists() and not files.db_path(root).exists() and (root / "vectors").is_dir()


def test_reset_refuses_foreign_dirs(tmp_path):
    (tmp_path / "precious.txt").write_text("x")
    with pytest.raises(ValueError):
        files.reset_data_dir(tmp_path, confirm=True)
    assert (tmp_path / "precious.txt").exists()
    with pytest.raises(ValueError):
        files.reset_data_dir("/", confirm=True)


def test_cli_info(tmp_path, capsys):
    files.ensure_dirs(tmp_path / "d")
    c = db.connect(files.db_path(tmp_path / "d"))
    db.create_run(c, "{}")
    c.close()
    assert (
        cli.main(["info", "--data-dir", str(tmp_path / "d"), "--set", "association.min_cos=0.8"])
        == 0
    )
    out = capsys.readouterr().out
    assert "min_cos: 0.8" in out and '"runs": 1' in out


def test_cli_app_arguments(capsys):
    """``reid app`` parses the flags the Colab notebook uses and validates --auth (no server)."""
    args = cli.build_parser().parse_args(
        ["app", "--share", "--auth", "reid:pw", "--preset", "balanced", "--data-dir", "x"]
    )
    assert args.share and args.auth == "reid:pw" and not args.read_only and args.func is cli.cmd_app
    assert cli.main(["app", "--auth", "nopassword"]) == 1
    assert "user:password" in capsys.readouterr().err


def test_cli_app_workspace_and_env(tmp_path):
    """--workspace/--env, and --data-dir X meaning workspace X.parent with environment X.name."""
    p = cli.build_parser()
    args = p.parse_args(["app", "--workspace", str(tmp_path), "--env", "own", "--read-only"])
    assert args.read_only and cli._app_workspace(args, cli._build_config(args)) == (tmp_path, "own")
    args = p.parse_args(["app", "--data-dir", str(tmp_path / "run1")])
    ws, env = cli._app_workspace(args, cli._build_config(args))
    assert ws == tmp_path.resolve() and env == "run1"
    args = p.parse_args(["app", "--set", f"paths.data_dir={tmp_path}", "--admin"])
    assert cli._app_workspace(args, cli._build_config(args)) == (tmp_path.resolve(), "default")
