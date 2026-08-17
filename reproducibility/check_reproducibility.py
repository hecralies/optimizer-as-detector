"""Validate the local environment, curated inputs, and real-data loaders."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path(__file__).with_name("data_manifest.json")
PACKAGES = [
    "numpy",
    "scipy",
    "pandas",
    "matplotlib",
    "scikit-learn",
    "numba",
    "seaborn",
    "openpyxl",
]


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def check_inputs() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    errors = []
    for entry in manifest["files"]:
        path = ROOT / entry["path"]
        if not path.is_file():
            errors.append(f"missing: {entry['path']}")
            continue
        if path.stat().st_size != entry["bytes"]:
            errors.append(f"size mismatch: {entry['path']}")
            continue
        if digest(path) != entry["sha256"]:
            errors.append(f"SHA-256 mismatch: {entry['path']}")
    if errors:
        raise RuntimeError("Input validation failed:\n  " + "\n  ".join(errors))
    print(f"Validated {len(manifest['files'])} curated input files.")


def check_packages() -> None:
    if sys.version_info < (3, 8):
        raise RuntimeError("Python 3.8 or newer is required.")
    print(f"Python {sys.version.split()[0]}")
    missing = []
    for package in PACKAGES:
        try:
            version = importlib.metadata.version(package)
            print(f"  {package}=={version}")
        except importlib.metadata.PackageNotFoundError:
            missing.append(package)
    if missing:
        raise RuntimeError("Missing packages: " + ", ".join(missing))


def check_loaders() -> None:
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    sys.path.insert(0, str(ROOT / "code"))
    import exp_realworld_K2_rigorous as real  # noqa: PLC0415

    print("Real-data loader audit:")
    for loader in real.LOADERS:
        dataset = loader()
        x = dataset["x"]
        n = len(dataset["y"])
        d = 1 if getattr(x, "ndim", 1) == 1 else x.shape[1]
        source = Path(str(dataset["source"]))
        if source.is_absolute() and ROOT not in source.parents:
            raise RuntimeError(
                f"{dataset['name']} resolved outside the project: {source}"
            )
        print(f"  {dataset['name']}: n={n}, d={d}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--loaders",
        action="store_true",
        help="also parse all eight manuscript datasets (slower, especially CarbonMonitor)",
    )
    args = parser.parse_args()
    check_packages()
    check_inputs()
    if args.loaders:
        check_loaders()
    print("Reproducibility checks passed.")


if __name__ == "__main__":
    main()
