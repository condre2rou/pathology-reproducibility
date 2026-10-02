"""Fit all annotation tasks with four isolated, single-threaded workers."""

import os

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "e11_method_audit"))
from data import load_tasks, manifest
from run import reference, operation, fingerprint, save


def compute(task, mode, destination):
    target = destination / mode / (task.name + ".json")
    if target.exists():
        raise FileExistsError("Fresh fitting must not reuse a per-task result: " + str(target))
    result = reference(task, 30) if mode == "reference" else operation(task, 30)
    result["fingerprint"] = fingerprint(task, SimpleNamespace(repeats=30), mode)
    save(target, result)
    return task.name


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["reference", "operation"])
    ap.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    args = ap.parse_args()
    tasks = load_tasks()
    destination = ROOT / "e11_method_audit/outputs"
    save(destination / "manifest.json", manifest(tasks))
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=mp.get_context("spawn")
    ) as executor:
        jobs = [executor.submit(compute, t, args.mode, destination) for t in tasks]
        for index, future in enumerate(as_completed(jobs), 1):
            print(f"{index}/{len(tasks)} {future.result()} freshly computed", flush=True)


if __name__ == "__main__":
    main()
