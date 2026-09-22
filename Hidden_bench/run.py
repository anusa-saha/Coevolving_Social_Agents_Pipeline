"""run.py - train, then evaluate, in one unattended command that survives crashes.

    python run.py

Runs `main.py` to completion and, ONLY if it exits 0, runs `inf.py` (vanilla vs coevolved
on data/1100_test.json: 180 seen + 200 unseen, with the per-domain tables). Nothing is
configurable from the command line; every knob lives in main.CONFIG / inf.CONFIG / gpu.py,
plus the three restart constants below.

Why a separate process per stage rather than importing both:
  main.py holds two models, two AdamW states and a LoRA graph on the GPUs for the whole
  run. Importing inf.py into the same interpreter would evaluate against whatever the
  allocator has already fragmented, and a stray reference to a training model would keep
  its weights pinned. A fresh process guarantees eval starts from an empty device.

CRASHES. Both stages checkpoint as they go (main.py every optimiser step, inf.py every
completed arm) and resume from the last commit on start. When a stage dies - OOM, a CUDA
error, a killed process - this script restarts it in a fresh process, which continues from
that commit. It gives up when:
  * the stage exits with resume.EXIT_FATAL (bad config/data, failed preflight, a checkpoint
    from a different experiment) - restarting cannot fix those;
  * MAX_STALLED restarts in a row made no checkpoint progress - a deterministic failure;
  * you press Ctrl+C.
Re-running `python run.py` at any point picks up exactly where the last commit left off.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

import resume

HERE = os.path.dirname(os.path.abspath(__file__))

MAX_ATTEMPTS = 50       # restarts per stage, in total
MAX_STALLED = 3         # consecutive crashes without checkpoint progress -> give up
RESTART_DELAY = 30      # seconds, so the driver reclaims the dead process's GPU memory

# Must match main.CONFIG.CKPT_DIR and inf.CONFIG.OUT_DIR (not imported: both pull in torch).
CKPT_LATEST = os.path.join(HERE, "ckpt", "latest")
EVAL_DIR = os.path.join(HERE, "results_inf")


def _fmt(secs: float) -> str:
    h, rem = divmod(int(secs), 3600)
    m, s = divmod(rem, 60)
    return "{}h {:02d}m {:02d}s".format(h, m, s) if h else "{}m {:02d}s".format(m, s)


def _progress(script: str):
    """A value that changes whenever the stage commits a checkpoint."""
    if script == "main.py":
        ck = resume.committed_dir(CKPT_LATEST)
        st = resume.read_json(os.path.join(ck, "state.json")) if ck else None
        st = st or {}
        return (st.get("round"), st.get("idx"), st.get("gstep"), st.get("done"))
    # inf.py commits per ARM: <tag>_done.json appears when that arm finished cleanly, and
    # its <tag>_scenarios.csv grows as blocks are written. Both move -> real progress.
    out = []
    if os.path.isdir(EVAL_DIR):
        for name in sorted(os.listdir(EVAL_DIR)):
            if name.endswith("_done.json"):
                out.append((name, resume.read_json(os.path.join(EVAL_DIR, name))))
            elif name.endswith("_scenarios.csv"):
                out.append((name, os.path.getsize(os.path.join(EVAL_DIR, name))))
    return tuple(out)


def _interrupted(rc: int) -> bool:
    return rc in (-signal.SIGINT, 130, 0xC000013A)     # POSIX signal, shell, Windows Ctrl+C


def stage(name: str, script: str) -> float:
    """Run one stage in its own interpreter until it succeeds. Raises SystemExit if it
    cannot."""
    banner = "=" * 78
    t0 = time.time()
    stalled = 0
    for attempt in range(1, MAX_ATTEMPTS + 1):
        print("\n{}\n>>> {}   ({})   attempt {}\n{}".format(
            banner, name, script, attempt, banner), flush=True)
        before = _progress(script)
        # -u so the child's stdout is unbuffered and `tee`/nohup see progress live.
        proc = subprocess.Popen([sys.executable, "-u", os.path.join(HERE, script)],
                                cwd=HERE)
        try:
            rc = proc.wait()
        except KeyboardInterrupt:
            proc.wait()
            raise SystemExit("\n!!! interrupted. Re-run `python run.py` to resume from the "
                             "last checkpoint.")

        if rc == 0:
            print("\n<<< {} finished in {}".format(name, _fmt(time.time() - t0)), flush=True)
            return time.time() - t0
        if rc == resume.EXIT_FATAL:
            raise SystemExit("\n{}\n!!! {} stopped on a FATAL error (see its log above). "
                             "Restarting cannot fix it.\n{}".format(banner, name, banner))
        if _interrupted(rc):
            raise SystemExit("\n!!! {} was interrupted. Re-run `python run.py` to resume "
                             "from the last checkpoint.".format(name))

        stalled = 0 if _progress(script) != before else stalled + 1
        if stalled >= MAX_STALLED:
            raise SystemExit(
                "\n{}\n!!! {} crashed {} times in a row without committing a checkpoint "
                "(last exit code {}).\n    That is a deterministic failure, not bad luck - "
                "fix it, then re-run `python run.py`;\n    it resumes from the last "
                "checkpoint.\n{}".format(banner, name, stalled, rc, banner))
        print("\n!!! {} exited with code {} after {}. Restarting from the last checkpoint "
              "in {}s ({} restart(s) without progress so far).".format(
                  name, rc, _fmt(time.time() - t0), RESTART_DELAY, stalled), flush=True)
        time.sleep(RESTART_DELAY)
    raise SystemExit("\n!!! {} did not finish in {} attempts.".format(name, MAX_ATTEMPTS))


def main() -> None:
    t0 = time.time()
    print("train_1100_big run.py   python {}".format(sys.version.split()[0]))
    print("working directory: {}".format(HERE))

    t_train = stage("TRAINING", "main.py")
    t_eval = stage("EVALUATION", "inf.py")

    print("\n" + "=" * 78)
    print("ALL DONE      train {}   eval {}   total {}".format(
        _fmt(t_train), _fmt(t_eval), _fmt(time.time() - t0)))
    print("  training artefacts : {}".format(os.path.join(HERE, "results_train")))
    print("  eval artefacts     : {}".format(EVAL_DIR))
    print("  checkpoints        : {}".format(os.path.join(HERE, "ckpt")))
    print("=" * 78)


if __name__ == "__main__":
    main()
