#!/usr/bin/env python3
"""Run manuscript experiments in a fresh, isolated output directory."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
E11 = "e11_method_audit/outputs"
E13 = "e13_review/outputs"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def stage(out, include_reference=True):
    if out.exists():
        raise FileExistsError("Output already exists; choose a new --out directory")
    work = out / "work"
    shutil.copytree(ROOT / "analysis", work, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    if include_reference:
        shutil.copytree(ROOT / "reference", work, dirs_exist_ok=True)
    for rel in (E11, E13, "figures", "e2_train/runs", "e9_revision", "e10_expansion"):
        (work / rel).mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir()
    return work


def run(out, work, arguments, environment=None):
    index = len(list((out / "logs").glob("*.log"))) + 1
    log = out / "logs" / f"{index:02d}_{Path(arguments[0]).stem}.log"
    env = os.environ.copy()
    env.update({key: "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")})
    env.update({"MPLBACKEND": "Agg", "PYTHONDONTWRITEBYTECODE": "1"})
    if environment:
        env.update(environment)
    print(f"[{index}] {' '.join(map(str, arguments))}", flush=True)
    with log.open("w") as stream:
        process = subprocess.run(
            [sys.executable, *map(str, arguments)],
            cwd=work,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    if process.returncode:
        raise RuntimeError(f"Command failed ({process.returncode}); see {log}")


def clear_derived(work):
    for rel in (
        f"{E11}/corrected_reference_result.json",
        f"{E11}/single_budget_result.json",
        f"{E11}/operational_auc_ci.json",
        f"{E13}/grouped_meta.json",
        f"{E13}/sensitivity.json",
    ):
        (work / rel).unlink(missing_ok=True)


def aggregate(out, work):
    clear_derived(work)
    for args in (
        ["e11_method_audit/summarize.py"],
        ["e11_method_audit/score_uncertainty.py"],
        ["e13_review/reanalyse.py", "cached"],
        ["e13_review/reanalyse.py", "summary"],
    ):
        run(out, work, args)


def reports(out, work, figures=True):
    run(out, work, [ROOT / "scripts/export_tables.py", "--root", work, "--out", out / "tables"])
    if figures:
        run(out, work, ["figures/make_figures.py"])
        names = json.loads((ROOT / "configs/experiment_map.json").read_text())["figures"]
        for item in names:
            for ext in ("pdf", "png", "svg"):
                source = work / "figures" / (item["stem"] + "." + ext)
                if not source.is_file():
                    raise FileNotFoundError(f"Expected figure was not generated: {source.name}")
                target = out / "figures" / (item["id"] + "." + ext)
                target.parent.mkdir(exist_ok=True)
                shutil.copy2(source, target)
    run(
        out,
        work,
        [
            ROOT / "scripts/compare_results.py",
            "--actual",
            work,
            "--reference",
            ROOT / "reference",
            "--out",
            out / "comparison.json",
        ],
    )


def copy_inputs(work, data_root, patterns):
    if not data_root or not data_root.is_dir():
        raise ValueError(
            "This mode requires --data-root with authorised input files; see docs/data.md"
        )
    selected = []
    for pattern in patterns:
        matches = sorted(data_root.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"Missing input pattern: {pattern} under --data-root")
        for source in matches:
            if not source.is_file():
                continue
            relative = source.relative_to(data_root)
            target = work / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            selected.append(
                {"file": str(relative), "bytes": source.stat().st_size, "sha256": sha(source)}
            )
    write(work.parent / "input_hashes.json", selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=["list", "figures", "aggregates", "example", "full", "supporting", "verify"]
    )
    parser.add_argument("--out", type=Path, help="New directory; existing output is never replaced")
    parser.add_argument(
        "--data-root", type=Path, help="Authorised cache root using docs/data.md layout"
    )
    parser.add_argument("--experiment", help="Supporting experiment ID from the list command")
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    parser.add_argument(
        "--private-images", type=Path, help="Original report/image root, when needed"
    )
    parser.add_argument("--hf-cache", type=Path, help="Existing Hugging Face model/feature cache")
    args = parser.parse_args()
    catalog = json.loads((ROOT / "configs/experiments.json").read_text())
    if args.mode == "list":
        print("Modes: figures, aggregates, example, full, supporting, verify\n")
        for key, item in catalog.items():
            print(f"{key:24s} {item['description']} [{item['access']}]")
        return
    if args.mode == "verify":
        subprocess.run([sys.executable, str(ROOT / "scripts/verify_release.py")], check=True)
        return
    if args.out is None:
        parser.error("--out is required for computations")
    if args.mode == "supporting" and args.experiment not in catalog:
        parser.error("Choose --experiment from the list command")
    if args.mode == "supporting" and args.experiment == "annotation":
        parser.error("Use full mode for the complete annotation and refitting experiment")
    if args.mode in ("full", "supporting") and args.data_root is None:
        parser.error("--data-root is required; no hospital inputs are bundled")
    out = args.out.resolve()
    work = stage(out)
    start = time.monotonic()
    status = {
        "mode": args.mode,
        "status": "running",
        "reference_results_are_not_refitting": args.mode in ("figures", "aggregates"),
    }
    write(out / "status.json", status)
    try:
        if args.mode == "figures":
            reports(out, work)
        elif args.mode == "aggregates":
            aggregate(out, work)
            reports(out, work)
        elif args.mode == "example":
            run(
                out,
                work,
                [
                    ROOT / "scripts/run_example.py",
                    "--root",
                    work,
                    "--data",
                    ROOT / "data/example/tcga_coad_msi_pool.npz",
                    "--out",
                    out / "example",
                ],
            )
        elif args.mode == "full":
            copy_inputs(work, args.data_root.resolve(), catalog["annotation"]["inputs"])
            for rel in (f"{E11}/reference", f"{E11}/operation", f"{E13}/reconstructed"):
                shutil.rmtree(work / rel)
            (work / E11 / "manifest.json").unlink()
            for mode in ("reference", "operation"):
                run(out, work, ["portable/fit_tasks.py", mode, "--workers", args.workers])
            clear_derived(work)
            run(out, work, ["e11_method_audit/summarize.py"])
            run(out, work, ["e11_method_audit/score_uncertainty.py"])
            run(out, work, ["e13_review/reanalyse.py", "refit"])
            run(out, work, ["e13_review/reanalyse.py", "cached"])
            run(out, work, ["e13_review/reanalyse.py", "summary"])
            reports(out, work)
        else:
            item = catalog[args.experiment]
            copy_inputs(work, args.data_root.resolve(), item["inputs"])
            env = {}
            if args.private_images:
                env["PATHOLOGY_PRIVATE_IMAGES"] = str(args.private_images.resolve())
            if args.hf_cache:
                env["PATHOLOGY_HF_CACHE"] = str(args.hf_cache.resolve())
                # Encoders are read from the supplied model cache, never bundled.
                (work / "models").symlink_to(args.hf_cache.resolve(), target_is_directory=True)
            for relative in item["results"]:
                (work / relative).unlink(missing_ok=True)
            for command in item["commands"]:
                run(out, work, command, env)
            run(
                out,
                work,
                [
                    ROOT / "scripts/compare_results.py",
                    "--actual",
                    work,
                    "--reference",
                    ROOT / "reference",
                    "--out",
                    out / "comparison.json",
                    "--files",
                    *item["results"],
                ],
            )
        status.update(status="complete", elapsed_seconds=round(time.monotonic() - start, 2))
    except Exception as exc:
        status.update(
            status="failed", error=str(exc), elapsed_seconds=round(time.monotonic() - start, 2)
        )
        write(out / "status.json", status)
        raise
    write(out / "status.json", status)
    print(f"Completed: {out}")


if __name__ == "__main__":
    main()
