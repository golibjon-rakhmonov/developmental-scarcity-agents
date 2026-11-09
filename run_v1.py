"""Runs every life of version 1 (see PREREGISTRATION_V1.md).

    .venv/bin/python run_v1.py --parallel 17
    .venv/bin/python run_v1.py --seeds 900 901 --out runs_v1_test      # any other seeds, e.g. to test
    .venv/bin/python run_v1.py --parallel 17 --seeds $(seq 140 169) --childhoods scarce hungry unreliable rich adult --out runs_exp2

Each process lives one childhood and then one adult life in each of the three adult worlds.
Lives that already finished are skipped, so the script can be stopped with one Ctrl+C and started
again. Results go to runs_v1/.
"""
import argparse
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

CHILDHOODS = ["scarce", "rich", "adult", "none"]
BRANCHES = ["standard", "richer", "poorer"]
NEW_MAP_EVERY = 1000
NEW_MAP_OFFSET = 500    # maps change halfway between drought starts and ends, not on them
SEEDS = range(100, 120)


def jobs(seeds, out, childhoods=CHILDHOODS):
    todo = []
    for s in seeds:
        for c in childhoods:
            if all((out / f"{c}_s{s}_{b}_end.pt").exists() for b in BRANCHES):
                continue
            cmd = [sys.executable, "run_life.py", "--childhood", c, "--seed", str(s), "--out", str(out),
                   "--new-map-every", str(NEW_MAP_EVERY), "--new-map-offset", str(NEW_MAP_OFFSET), "--branches", *BRANCHES]
            todo.append((f"{c}_s{s}", cmd))
    # the longest lives first, so the last minutes are not spent waiting for one slow process
    return sorted(todo, key=lambda j: j[0].startswith("none"))


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
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    p.add_argument("--childhoods", nargs="+", default=CHILDHOODS,
                   choices=["scarce", "rich", "adult", "unreliable", "hungry", "none"],
                   help="experiment 2 adds the unreliable and the hungry childhood; the default is version 1")
    p.add_argument("--out", default="runs_v1")
    args = p.parse_args()
    out = Path(args.out)
    todo = jobs(args.seeds, out, args.childhoods)
    print(f"{len(todo)} lives to run, {args.parallel} at a time", flush=True)
    started, failed = time.time(), 0
    pool = ThreadPoolExecutor(args.parallel)
    try:
        futures = [pool.submit(run, name, cmd) for name, cmd in todo]
        for k, f in enumerate(as_completed(futures), 1):
            name, code, secs, err = f.result()
            elapsed = (time.time() - started) / 60
            left = elapsed / k * (len(todo) - k)
            # a negative code means the process was killed by a signal (for example out of memory)
            status = "done" if code == 0 else "FAILED" if code > 0 else f"KILLED (signal {-code})"
            print(f"[{k}/{len(todo)}] {status} {name} in {secs / 60:.1f} min | {elapsed:.0f} min so far, about {left:.0f} left", flush=True)
            if code != 0:
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
    if failed:
        print(f"finished, {failed} failed or killed. Run the same command again to redo them.", flush=True)
    else:
        print("finished, none failed. Next: .venv/bin/python analyze_v1.py", flush=True)


if __name__ == "__main__":
    main()
