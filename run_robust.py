"""Runs every life of the robustness check (see PREREGISTRATION_ROBUSTNESS.md).

    .venv/bin/python run_robust.py                 everything, 8 lives at a time
    .venv/bin/python run_robust.py --parallel 10
    .venv/bin/python run_robust.py --only replication

Lives that already finished are skipped, so the script can be stopped with
one Ctrl+C and started again. Results go to runs_robust/<variant>/.
"""
import argparse
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path("runs_robust")
SEEDS_20, SEEDS_10 = range(10, 30), range(10, 20)

# variant: (childhoods, seeds, extra arguments for run_life.py)
VARIANTS = {
    "replication": (["scarce", "rich", "none"], SEEDS_20, []),
    "long_childhood": (["scarce", "rich"], SEEDS_10, ["--child-steps", "6000000"]),
    "short_childhood": (["scarce", "rich"], SEEDS_10, ["--child-steps", "1000000"]),
    "scarcer": (["scarce", "rich"], SEEDS_10, ["--scarce-regrow", "0.0007"]),
    "less_scarce": (["scarce", "rich"], SEEDS_10, ["--scarce-regrow", "0.002"]),
    "reset": (["scarce", "rich"], SEEDS_10, ["--reset", "last"]),
}


def jobs(only):
    out = []
    for variant, (childhoods, seeds, extra) in VARIANTS.items():
        if only and variant not in only:
            continue
        folder = ROOT / variant
        for s in seeds:
            for c in childhoods:
                if (folder / f"{c}_s{s}_end.pt").exists():
                    continue
                cmd = [sys.executable, "run_life.py", "--childhood", c, "--seed", str(s), "--out", str(folder), *extra]
                out.append((f"{variant}/{c}_s{s}", cmd))
    return out


RUNNING, LOCK = set(), threading.Lock()


def run(name, cmd):
    started = time.time()
    # own session, so Ctrl+C reaches only this script, which then stops the lives itself
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, start_new_session=True)
    with LOCK:
        RUNNING.add(proc)
    _, err = proc.communicate()
    with LOCK:
        RUNNING.discard(proc)
    return name, proc.returncode, time.time() - started, (err or "")[-800:]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parallel", type=int, default=8)
    p.add_argument("--only", nargs="*", choices=list(VARIANTS))
    args = p.parse_args()
    todo = jobs(args.only)
    print(f"{len(todo)} lives to run, {args.parallel} at a time", flush=True)
    started, failed = time.time(), 0
    pool = ThreadPoolExecutor(args.parallel)
    try:
        futures = [pool.submit(run, name, cmd) for name, cmd in todo]
        for k, f in enumerate(as_completed(futures), 1):
            name, code, secs, err = f.result()
            elapsed = (time.time() - started) / 60
            left = elapsed / k * (len(todo) - k)
            status = "done" if code == 0 else "FAILED"
            print(f"[{k}/{len(todo)}] {status} {name} in {secs / 60:.1f} min | {elapsed:.0f} min so far, about {left:.0f} left", flush=True)
            if code != 0 and code > 0:
                failed += 1
                print(err, flush=True)
    except KeyboardInterrupt:
        pool.shutdown(wait=False, cancel_futures=True)
        with LOCK:
            for proc in RUNNING:
                proc.kill()
        print("\nstopped. Finished lives are kept; run the same command again to continue.", flush=True)
        os._exit(130)
    pool.shutdown()
    print(f"finished, {failed} failed. Next: .venv/bin/python analyze_robust.py", flush=True)


if __name__ == "__main__":
    main()
