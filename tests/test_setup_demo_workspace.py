"""scripts/setup_demo_workspace.py: import zips, copy a plain env, install the head, annotate."""

import importlib.util
import json
from pathlib import Path

from reid.attributes.head import save_head
from reid.storage import attributes_db as adb
from reid.storage import db, files, portable
from tests.attr_synth import make_head
from tests.test_attr_annotate import _make_env_dir

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "setup_demo_workspace.py"


def _load():
    spec = importlib.util.spec_from_file_location("setup_demo_workspace", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_setup_is_idempotent_and_annotates(tmp_path, monkeypatch):
    mod = _load()
    src, conn = _make_env_dir(tmp_path / "src" / "envA")
    conn.close()
    portable.export_env(src, tmp_path / "zips" / "envA.zip")
    plain, conn = _make_env_dir(tmp_path / "plain" / "warehouse")
    conn.close()
    (plain / "env.json").unlink()
    (plain / "renders").mkdir(exist_ok=True)
    (plain / "renders" / "big.mp4").write_bytes(b"x")
    save_head(tmp_path / "head", make_head())
    ws = tmp_path / "ws"
    argv = [
        "--workspace", str(ws), "--head", str(tmp_path / "head"),
        "--zips", str(tmp_path / "zips" / "*.zip"), "--warehouse", str(plain),
        "--weights-dir", str(tmp_path / "w"),
    ]  # fmt: skip
    monkeypatch.setattr(mod, "ROOT", Path("/"))
    assert mod.main(argv) == 0
    assert sorted(p.name for p in ws.iterdir()) == ["envA", "warehouse"]
    assert not (ws / "warehouse" / "renders").exists() and (ws / "warehouse" / "env.json").exists()
    assert (tmp_path / "w" / "attributes" / "head.npz").is_file()
    for name in ("envA", "warehouse"):
        c = db.connect(files.db_path(ws / name))
        assert adb.count_annotated(c) == 1
        c.close()
    assert mod.main(argv) == 0  # second run changes nothing
    assert json.loads((ws / "envA" / "env.json").read_text())["name"] == "envA"
