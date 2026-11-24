"""Execute notebooks/colab_demo.ipynb locally, the way "Run all" does on Colab.

Cells tagged ``colab-only`` (pip installs, the blocking app launch) are skipped, and the
``--set`` overrides are injected right after the ``settings`` cell, for example::

    python scripts/test_notebook.py --out /tmp/nb_aicity.ipynb \\
        --set SOURCE='"aicity"' --set AICITY_MINUTES=1 --set LOCAL_DATA_DIR='"data/nbtest"'

Values are Python expressions. The executed notebook (with outputs) is written to ``--out`` so the
output of every cell can be inspected; the exit code is non-zero if a cell raised.
Needs ``nbclient`` and ``ipykernel`` (the ``dev`` extra) in the environment running the package.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "colab_demo.ipynb"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--notebook", type=Path, default=NOTEBOOK)
    ap.add_argument("--out", type=Path, required=True, help="executed notebook (with outputs)")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=EXPR")
    ap.add_argument("--timeout", type=int, default=3600, help="per-cell timeout in seconds")
    ap.add_argument("--cwd", type=Path, default=Path.cwd(), help="working directory of the kernel")
    args = ap.parse_args()

    nb = nbformat.read(args.notebook, as_version=4)
    cells = []
    for cell in nb.cells:
        tags = cell.get("metadata", {}).get("tags", [])
        if "colab-only" in tags:
            continue
        cells.append(cell)
        if "settings" in tags and args.set:
            src = "# injected by scripts/test_notebook.py\n" + "\n".join(args.set)
            cells.append(nbformat.v4.new_code_cell(src))
    nb.cells = cells

    client = NotebookClient(
        nb, timeout=args.timeout, kernel_name="python3", resources={"metadata": {"path": args.cwd}}
    )
    failed = False
    try:
        client.execute()
    except CellExecutionError as exc:
        failed = True
        print(f"FAILED:\n{exc}", file=sys.stderr)
    finally:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(nb, args.out)
        print(f"executed notebook written to {args.out}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
