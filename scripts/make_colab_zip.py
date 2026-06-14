"""
make_colab_zip.py — build paut-digital-twin.zip for Colab/Kaggle upload.

Uses Python's zipfile with FORWARD-SLASH archive names so it extracts correctly
on Linux (Colab). PowerShell's Compress-Archive writes backslash separators,
which Linux unzip treats as part of the filename instead of folders — that breaks
`os.chdir('paut-digital-twin')` in the notebook.

Includes code + data/raw; excludes processed data, checkpoints, caches.

Run from the project root:
    py -3.14 scripts/make_colab_zip.py
"""

from __future__ import annotations

import os
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOP = "paut-digital-twin"                      # top-level folder inside the zip
OUT = os.path.join(ROOT, "paut-digital-twin.zip")

INCLUDE_DIRS = ["src", "configs", "tests", "notebooks", "scripts", os.path.join("data", "raw")]
INCLUDE_FILES = ["design_decisions.md", "README.md", "requirements.txt", ".gitignore"]
SKIP_PARTS = {"__pycache__", ".ipynb_checkpoints"}


def _keep(path: str) -> bool:
    parts = set(path.replace("\\", "/").split("/"))
    if parts & SKIP_PARTS:
        return False
    return not path.endswith(".pyc")


def main():
    if os.path.exists(OUT):
        os.remove(OUT)
    n = 0
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for f in INCLUDE_FILES:
            p = os.path.join(ROOT, f)
            if os.path.exists(p):
                z.write(p, f"{TOP}/{f}")               # forward slashes
                n += 1
        for d in INCLUDE_DIRS:
            base = os.path.join(ROOT, d)
            if not os.path.isdir(base):
                continue
            for cur, dirs, files in os.walk(base):
                dirs[:] = [x for x in dirs if x not in SKIP_PARTS]
                for fn in files:
                    full = os.path.join(cur, fn)
                    rel = os.path.relpath(full, ROOT).replace("\\", "/")
                    if _keep(rel):
                        z.write(full, f"{TOP}/{rel}")
                        n += 1
    size_mb = os.path.getsize(OUT) / 1e6
    print(f"wrote {OUT}")
    print(f"  {n} files, {size_mb:.1f} MB")
    # sanity: confirm forward slashes and a real top-level folder
    with zipfile.ZipFile(OUT) as z:
        names = z.namelist()
        assert all("\\" not in nm for nm in names), "backslash in zip entry!"
        assert any(nm.startswith(f"{TOP}/data/raw/") for nm in names), "data/raw missing!"
    print("  OK: forward-slash paths, data/raw present")


if __name__ == "__main__":
    main()
