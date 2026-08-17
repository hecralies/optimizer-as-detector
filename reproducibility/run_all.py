"""Run the JASA reproducibility pipeline without overwriting paper artifacts."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
ALL_STAGES = ("synthetic", "robustness", "realdata", "benchmarks", "figures")


def command_specs(mode: str, stages: set[str], run_dir: Path):
    py = sys.executable
    results = run_dir / "results"
    figures = run_dir / "fig"
    quick = ["--quick"] if mode == "quick" else []
    specs: list[tuple[str, list[str]]] = []

    if "synthetic" in stages:
        specs.extend(
            [
                ("synthetic-null", [py, "code/exp_4pi_universality.py", *quick, "--out-dir", str(results)]),
                ("synthetic-alternative", [py, "code/exp_synthetic_h1_robust.py", *quick, "--out-dir", str(results), "--fig-dir", str(figures)]),
                ("fresh-stream-size", [py, "code/exp_type1_rigorous.py", *quick, "--out-dir", str(results)]),
                ("fresh-stream-power", [py, "code/exp_power_H1_rigorous.py", *quick, "--out-dir", str(results)]),
            ]
        )

    if "robustness" in stages:
        specs.extend(
            [
                ("heteroskedasticity", [py, "code/exp_heterosked_fix_verify.py", *quick, *(["--n-data", "5000", "--t-sgd", "5000", "--n-trials", "3"] if mode == "quick" else []), "--out-dir", str(results)]),
                ("misspecification", [py, "code/exp_misspecification.py", *quick, *(["--n-data", "5000", "--t-sgd", "5000", "--n-trials", "3"] if mode == "quick" else []), "--out-dir", str(results)]),
                ("finite-sample-reuse", [py, "code/exp_finite_sample_reuse.py", *quick, "--out-dir", str(results)]),
            ]
        )

    if "realdata" in stages:
        if mode == "quick":
            specs.extend(
                [
                    ("real-stage1", [py, "code/exp_realworld_robust.py", "--project-root", str(ROOT), "--out-dir", str(results), "--quick", "--datasets", "Iris"]),
                    ("binary-remedy", [py, "code/exp_categorical_pooled_remedy.py", "--project-root", str(ROOT), "--out-dir", str(results), "--passes", "5", "--seeds", "2", "--datasets", "Taylor"]),
                    ("real-screen", [py, "code/exp_realworld_K2_rigorous.py", "--quick", "--out-dir", str(results), "--datasets", "Iris"]),
                ]
            )
        else:
            specs.extend(
                [
                    ("real-stage1", [py, "code/exp_realworld_robust.py", "--project-root", str(ROOT), "--out-dir", str(results)]),
                    ("binary-remedy", [py, "code/exp_categorical_pooled_remedy.py", "--project-root", str(ROOT), "--out-dir", str(results)]),
                    ("real-screen", [py, "code/exp_realworld_K2_rigorous.py", "--out-dir", str(results)]),
                ]
            )

    if "benchmarks" in stages:
        if mode == "quick":
            specs.extend(
                [
                    ("ks-em", [py, "code/exp_kasahara_shimotsu_em_nine.py", "--out-dir", str(results), "--datasets", "Iris", "--boot", "3", "--restarts", "2", "--max-iter", "15", "--suffix", "quick"]),
                    ("ks-em-ca-full-x", [py, "code/exp_kasahara_shimotsu_em_nine.py", "--out-dir", str(results), "--datasets", "CA Housing", "--full-x", "--boot", "0", "--restarts", "1", "--max-iter", "5", "--suffix", "ca_fullx_quick"]),
                    ("em-bic", [py, "code/exp_conventional_em_detection_nine.py", "--out-dir", str(results), "--datasets", "Iris", "--boot", "0", "--observed-restarts", "2", "--max-iter", "30", "--suffix", "quick"]),
                ]
            )
        else:
            specs.extend(
                [
                    ("ks-em", [py, "code/exp_kasahara_shimotsu_em_nine.py", "--out-dir", str(results), "--restarts", "4", "--max-iter", "40"]),
                    ("ks-em-ca-full-x", [py, "code/exp_kasahara_shimotsu_em_nine.py", "--out-dir", str(results), "--datasets", "CA Housing", "--full-x", "--restarts", "4", "--max-iter", "40", "--suffix", "ca_fullx"]),
                    ("em-bic", [py, "code/exp_conventional_em_detection_nine.py", "--out-dir", str(results), "--boot", "0", "--suffix", "em_bic_final"]),
                ]
            )

    if "figures" in stages:
        specs.extend(
            [
                ("figure2-ab", [py, "code/fig_synthetic_summary.py", "--results-dir", str(results), "--fig-dir", str(figures)]),
                ("figure2-cd", [py, "code/fig_heterosked_misspec.py", "--results-dir", str(results), "--fig-dir", str(figures)]),
                ("universality-figure", [py, "code/fig_4pi_universality.py", "--results-dir", str(results), "--fig-dir", str(figures)]),
            ]
        )
        if mode == "full":
            specs.extend(
                [
                    ("figure1", [py, "code/fig1_ensemble_ode.py", "--fig-dir", str(figures)]),
                    ("higher-k", [py, "code/fig_K34_H0_H1.py", "--fig-dir", str(figures)]),
                ]
            )
    return specs


def run_command(name: str, argv: list[str], log_dir: Path) -> dict[str, object]:
    print("\n$ " + " ".join(argv), flush=True)
    started = time.time()
    log_path = log_dir / f"{name}.log"
    with log_path.open("w", encoding="utf-8") as log:
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        if "--quick" in argv:
            environment["NUMBA_DISABLE_JIT"] = "1"
        process = subprocess.Popen(
            argv,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=environment,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return_code = process.wait()
    record = {
        "name": name,
        "argv": argv,
        "return_code": return_code,
        "elapsed_seconds": time.time() - started,
        "log": str(log_path.relative_to(ROOT)),
    }
    if return_code:
        raise subprocess.CalledProcessError(return_code, argv)
    return record


def environment_record() -> dict[str, object]:
    packages = {}
    for name in ("numpy", "scipy", "pandas", "matplotlib", "scikit-learn", "numba"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--quick", action="store_true")
    mode.add_argument("--full", action="store_true")
    parser.add_argument("--stages", nargs="+", choices=ALL_STAGES, default=list(ALL_STAGES))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    selected_mode = "full" if args.full else "quick" if args.quick else "check"
    check_command = [sys.executable, str(HERE / "check_reproducibility.py")]
    if selected_mode == "check":
        subprocess.run(check_command, cwd=ROOT, check=True)
        return

    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = (args.output_dir or HERE / "runs" / f"{stamp}-{selected_mode}").resolve()
    specs = command_specs(selected_mode, set(args.stages), run_dir)

    if args.dry_run:
        print("$ " + " ".join(check_command))
        for _, argv in specs:
            print("$ " + " ".join(argv))
        print(f"Outputs: {run_dir}")
        return

    (run_dir / "results").mkdir(parents=True, exist_ok=True)
    (run_dir / "fig").mkdir(parents=True, exist_ok=True)
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(check_command, cwd=ROOT, check=True)
    manifest = {
        "mode": selected_mode,
        "stages": args.stages,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "root": str(ROOT),
        "run_dir": str(run_dir),
        "environment": environment_record(),
        "commands": [],
    }
    manifest_path = run_dir / "run_manifest.json"
    try:
        for name, argv in specs:
            manifest["commands"].append(run_command(name, argv, log_dir))
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    finally:
        manifest["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nReproduction run complete: {run_dir}")


if __name__ == "__main__":
    main()
