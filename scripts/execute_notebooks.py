"""Re-execute notebooks via papermill, then tailor §9 commentary.

Usage:
    python scripts/execute_notebooks.py                    # all 35
    python scripts/execute_notebooks.py --only 18,19,20    # specific ids
    python scripts/execute_notebooks.py --skip-build       # don't re-run build_NN_*.py

For each notebook NN:
  1. (optional) run scripts/build_NN_<name>.py — regenerates the .ipynb cells
  2. papermill notebooks/NN_*.ipynb in-place using the python3 kernel
  3. run scripts/tailor_NN_commentary.py — rewrites §9 from captured output

Exits non-zero if any step fails.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NOTEBOOKS = REPO / "notebooks"
SCRIPTS = REPO / "scripts"

# Match e.g. "18_reflexion.ipynb" → ("18", "reflexion")
NB_PATTERN = re.compile(r"^(\d{2})_([a-z0-9_]+)\.ipynb$")


def discover_notebooks() -> list[tuple[str, str, Path]]:
    out: list[tuple[str, str, Path]] = []
    for p in sorted(NOTEBOOKS.glob("*.ipynb")):
        m = NB_PATTERN.match(p.name)
        if m:
            out.append((m.group(1), m.group(2), p))
    return out


def run(cmd: list[str], **kw: object) -> int:
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, **kw).returncode  # type: ignore[arg-type]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="Comma-separated NN ids; empty = all")
    ap.add_argument("--skip-build", action="store_true", help="Don't run build_NN_*.py")
    ap.add_argument("--skip-tailor", action="store_true", help="Don't run tailor_NN_*.py")
    args = ap.parse_args()

    only = {s.strip() for s in args.only.split(",") if s.strip()}
    nbs = discover_notebooks()
    if only:
        nbs = [t for t in nbs if t[0] in only]

    if not nbs:
        print(f"No notebooks matched filter {only!r}")
        return 1

    print(f"Re-executing {len(nbs)} notebooks: {[t[0] for t in nbs]}")
    failed: list[str] = []

    for nn, name, path in nbs:
        print(f"\n=== [{nn}] {name} ===")

        build_script = SCRIPTS / f"build_{nn}_{name}.py"
        if not args.skip_build and build_script.exists():
            rc = run([sys.executable, str(build_script)])
            if rc != 0:
                failed.append(f"build_{nn}_{name}.py rc={rc}")
                continue

        rc = run([
            sys.executable, "-m", "papermill",
            str(path), str(path),
            "--kernel", "python3",
        ])
        if rc != 0:
            failed.append(f"papermill {path.name} rc={rc}")
            continue

        tailor_script = SCRIPTS / f"tailor_{nn}_commentary.py"
        if not args.skip_tailor and tailor_script.exists():
            rc = run([sys.executable, str(tailor_script)])
            if rc != 0:
                failed.append(f"tailor_{nn}_commentary.py rc={rc}")
                continue

    print(f"\n{'=' * 60}")
    if failed:
        print(f"FAILED ({len(failed)}/{len(nbs)}):")
        for f in failed:
            print(f"  - {f}")
        return 1
    print(f"OK: {len(nbs)}/{len(nbs)} notebooks refreshed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
