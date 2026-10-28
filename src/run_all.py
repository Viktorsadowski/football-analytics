#!/usr/bin/env python3
"""
Runs the whole chain in order. each step's output goes to logs/<step>.log, the console gets one line per step.

order notes (to remember):
  - box_prior.py turns rapm.py's plain rapm.parquet into the priored one (plain kept as rapm_plain.parquet),
    so rapm.py has to run right before it, every time
  - war.py runs twice: league_strength.py needs war.parquet, and war.py puts league strength into the values
  - statsbomb_pull.py downloads, so it's skipped when its output is already there (--pull to force)
  - one thread for numpy / BLAS: multithreaded sums differ in the last digits from run to run, and the united
    case's greedy search can flip on a near tie (A picked barcola one run, ali-cho the next)

  python src/run_all.py                 # everything
  python src/run_all.py --from war      # from the first war.py on
"""

import os
import subprocess
import sys
import time
from pathlib import Path

from config import RAW_DIR, ROOT

STEPS = ["ingest", "segments", "rapm", "box_prior", "tm_match", "gk", "aging", "elo", "inferred_injuries",
         "injury", "availability", "war", "league_strength", "war", "price", "market_value", "statsbomb_pull",
         "event_stats", "defense_study", "side_questions", "forecast_backtest", "wages", "transfer_value",
         "united_case", "season_sim"]


def main():
    args = sys.argv[1:]
    start = args[args.index("--from") + 1] if "--from" in args else STEPS[0]
    steps = STEPS[STEPS.index(start):]
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    seen = {}
    t_all = time.time()
    for s in steps:
        if s == "statsbomb_pull" and "--pull" not in args and (RAW_DIR / "statsbomb" / "sb_2015_player_match.parquet").exists():
            print(f"  {s:20s} skipped (already pulled)")
            continue
        seen[s] = seen.get(s, 0) + 1
        log = logs / (f"{s}.log" if seen[s] == 1 else f"{s}_{seen[s]}.log")
        t = time.time()
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        with open(log, "w") as f:
            r = subprocess.run([sys.executable, str(ROOT / "src" / f"{s}.py")], stdout=f, stderr=subprocess.STDOUT,
                               cwd=ROOT, env=env)
        print(f"  {s:20s} {'ok' if r.returncode == 0 else 'FAILED'}  {time.time() - t:6.0f}s  -> {log.relative_to(ROOT)}",
              flush=True)
        if r.returncode != 0:
            print(Path(log).read_text()[-2000:])
            sys.exit(1)
    print(f"done in {(time.time() - t_all) / 60:.0f} min")


if __name__ == "__main__":
    main()
