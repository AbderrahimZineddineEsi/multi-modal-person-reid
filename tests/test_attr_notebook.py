"""The Kaggle attributes notebook is generated, has no outputs and keeps numpy out of the kernel."""

import ast
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
NB = ROOT / "notebooks" / "kaggle_attributes.ipynb"


def _builder():
    spec = importlib.util.spec_from_file_location(
        "build_kaggle_attributes_notebook", ROOT / "scripts" / "build_kaggle_attributes_notebook.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_notebook_matches_builder_and_has_no_outputs(tmp_path):
    mod = _builder()
    mod.NB_PATH = tmp_path / "nb.ipynb"
    mod.write()
    assert json.loads(mod.NB_PATH.read_text()) == json.loads(NB.read_text()), "run the builder"
    nb = json.loads(NB.read_text())
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert code and all(c["outputs"] == [] and c["execution_count"] is None for c in code)
    tags = {t for c in code for t in c["metadata"].get("tags", [])}
    assert tags == {"settings", "kaggle-only"}
    text = "".join("".join(c["source"]) for c in code)
    for cmd in ("fetch", "embed", "train", "report"):
        assert f'"reid", "attributes", "{cmd}"' in text


def test_notebook_kernel_imports_only_stdlib():
    allowed = {"reid.notebook", "IPython.display"}
    for cell in json.loads(NB.read_text())["cells"]:
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
                assert name in allowed or name.split(".")[0] in sys.stdlib_module_names, name


HPO_NB = ROOT / "notebooks" / "kaggle_attributes_hpo.ipynb"


def test_hpo_notebook_matches_builder_and_has_no_outputs(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "build_kaggle_attributes_hpo_notebook",
        ROOT / "scripts" / "build_kaggle_attributes_hpo_notebook.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.NB_PATH = tmp_path / "nb.ipynb"
    mod.write()
    assert json.loads(mod.NB_PATH.read_text()) == json.loads(HPO_NB.read_text()), "run the builder"
    nb = json.loads(HPO_NB.read_text())
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert code and all(c["outputs"] == [] and c["execution_count"] is None for c in code)
    assert {t for c in code for t in c["metadata"].get("tags", [])} == {"settings", "kaggle-only"}
    text = "".join("".join(c["source"]) for c in code)
    for cmd in ("hpo", "hpo-final"):
        assert f'"reid", "attributes", "{cmd}"' in text
    assert "TRIAL |" in text and "reid[train]" in text and "reid[models" not in text
    for cell in nb["cells"]:  # every code cell parses (cell magics aside)
        src = "".join(cell["source"])
        if cell["cell_type"] == "code" and not src.lstrip().startswith("%%"):
            ast.parse(src)
            for node in ast.walk(ast.parse(src)):
                names = [a.name for a in node.names] if isinstance(node, ast.Import) else []
                if isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                for name in names:
                    assert name in {"reid.notebook", "IPython.display"} or (
                        name.split(".")[0] in sys.stdlib_module_names
                    ), name
