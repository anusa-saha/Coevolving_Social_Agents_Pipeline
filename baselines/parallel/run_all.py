"""Run every baseline's pipeline across the GPUs of one box, in parallel.

    python parallel/run_all.py --dry_run          # the plan and an estimated timeline
    python parallel/run_all.py                    # run it; re-running resumes
    python parallel/run_all.py --only sr om       # some arms (their deps come along)
    python parallel/run_all.py --status           # where the current or last run is

Built for 4 x 24 GB (A5000). One Qwen3.5-9B is ~17 GiB of bf16 weights, so a card holds one
model and runs one job; the only job that takes two cards is EPO's RL stage (frozen agent
on one, trained strategist on the other). Each job is pinned with CUDA_VISIBLE_DEVICES, so
inside it the devices are always cuda:0 (and cuda:1).

Scheduling. A job starts when its dependencies have succeeded and enough GPUs are free.
Ready jobs are ordered by the longest chain of work still ahead of them (the job plus its
longest line of dependents), priority breaking ties: the wall clock is set by the longest
chain, so it has to start first. A two-GPU job that
cannot fit reserves the moment enough running jobs are expected to end (EASY backfilling):
lower-priority jobs may take the idle cards meanwhile, but only if they are expected to
finish before that moment. Without that, a ready two-GPU job either starves or holds a
card idle for a day.

Resuming. A job that exits 0 leaves <run_dir>/state/<job>.done and is not run again
(--reset JOB to redo it). Anything else re-runs on the next launch; the collection and
corpus stages pick up from the scenarios they already wrote.

Failures do not stop the run. The failed job's dependents are marked blocked, everything
else carries on, and the final table says which is which. Logs: <run_dir>/logs/<job>.log.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import jobs as J                                     # noqa: E402

PENDING, RUNNING, DONE = 'pending', 'running', 'done'
FAILED, BLOCKED, GATED, INTERRUPTED = 'failed', 'blocked', 'gated', 'interrupted'
FINISHED = (DONE, FAILED, BLOCKED, GATED)
NOT_OK = (FAILED, BLOCKED, GATED)
HEADER = '#################### run_all attempt '


# ====================================================================== helpers
def detect_gpus():
    """[(index, name, MiB)] from nvidia-smi, or [] when there is none."""
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total',
                              '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return []
    gpus = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(',')]
        if len(parts) == 3 and parts[0].isdigit():
            gpus.append((int(parts[0]), parts[1], int(float(parts[2]))))
    return gpus


def validate(jobs):
    by = {}
    for j in jobs:
        if j.name in by:
            raise SystemExit('duplicate job %s' % j.name)
        by[j.name] = j
    for j in jobs:
        for d in j.deps + j.after:
            if d not in by:
                raise SystemExit('%s depends on unknown job %s' % (j.name, d))
    seen, stack = {}, []

    def visit(n):
        if seen.get(n) == 1:
            raise SystemExit('dependency cycle: %s' % ' -> '.join(stack + [n]))
        if seen.get(n) == 2:
            return
        seen[n] = 1
        stack.append(n)
        for d in by[n].deps + by[n].after:
            visit(d)
        stack.pop()
        seen[n] = 2

    for j in jobs:
        visit(j.name)
    return by


def select(jobs, only):
    """The requested jobs plus everything they depend on."""
    if not only:
        return list(jobs)
    by = {j.name: j for j in jobs}
    want = {j.name for j in jobs
            if j.arm in only or j.name in only
            or any(j.name.startswith(o.rstrip('.') + '.') for o in only)}
    if not want:
        raise SystemExit('--only matched nothing; arms are: %s'
                         % ', '.join(sorted({j.arm for j in jobs})))
    stack = list(want)
    while stack:
        for d in by[stack.pop()].deps:
            if d not in want:
                want.add(d)
                stack.append(d)
    return [j for j in jobs if j.name in want]


def assign_tails(jobs):
    """j.tail = j.hours + the longest chain of jobs that wait on j (critical path)."""
    children = {j.name: [] for j in jobs}
    for j in jobs:
        for d in j.deps + j.after:
            if d in children:
                children[d].append(j)
    memo = {}

    def tail(j):
        if j.name not in memo:
            memo[j.name] = j.hours + max([tail(c) for c in children[j.name]] or [0.0])
        return memo[j.name]

    for j in jobs:
        j.tail = tail(j)


def _rank(j):
    return (-getattr(j, 'tail', 0.0), -j.priority)


def plan_starts(now, ready, free, running, max_cpu):
    """Which ready jobs to start now, and on which GPUs.

    `running` is [(job, gpus, expected_end_hours)]. Longest chain first. When a
    multi-GPU job cannot fit, it holds a reservation at the time enough running jobs are
    expected to end; lower-priority jobs may then take idle GPUs only if they are expected
    to finish before that time, or use GPUs the reservation will not need.
    """
    starts = []
    cpu_busy = sum(1 for j, _g, _e in running if not j.gpus)
    for j in sorted((j for j in ready if not j.gpus), key=_rank):
        if cpu_busy >= max_cpu:
            break
        starts.append((j, []))
        cpu_busy += 1

    free = list(free)
    ends = [(e, len(g)) for j, g, e in running if j.gpus]
    shadow = spare = None
    for j in sorted((j for j in ready if j.gpus), key=_rank):
        if j.gpus > len(free):
            if shadow is None and j.gpus > 1:
                avail, shadow, spare = len(free), float('inf'), 0
                for e, n in sorted(ends):
                    avail += n
                    if avail >= j.gpus:
                        shadow, spare = e, avail - j.gpus
                        break
            continue
        if shadow is not None and now + j.hours > shadow:
            if j.gpus > spare:
                continue
            spare -= j.gpus
        take, free = free[:j.gpus], free[j.gpus:]
        starts.append((j, take))
        ends.append((now + j.hours, j.gpus))
    return starts


def load_durations(run_dir):
    try:
        with open(os.path.join(run_dir, 'durations.json'), encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def fmt_h(h):
    return '-' if h is None else '%.1f' % h


# ====================================================================== dry run
def dry_run(jobs, gpus, max_cpu, done):
    """Event simulation of the schedule on the estimated durations."""
    status = {j.name: (DONE if j.name in done else PENDING) for j in jobs}
    names = {j.name for j in jobs}
    free, running, rows, now = list(gpus), [], [], 0.0
    while True:
        ready = [j for j in jobs if status[j.name] == PENDING
                 and all(status[d] == DONE for d in j.deps)
                 and all(status.get(a, DONE) in FINISHED for a in j.after if a in names)]
        starts = plan_starts(now, ready, free, [(j, g, e) for j, g, _s, e in running],
                             max_cpu)
        for j, g in starts:
            status[j.name] = RUNNING
            for x in g:
                free.remove(x)
            running.append((j, g, now, now + j.hours))
            rows.append((j, g, now, now + j.hours))
        if not running:
            break
        now = min(e for _j, _g, _s, e in running)
        for item in [r for r in running if r[3] <= now + 1e-9]:
            running.remove(item)
            status[item[0].name] = DONE
            free = sorted(free + item[1])

    print('%-22s %-6s %8s %8s %7s  %s' % ('job', 'gpus', 'start h', 'end h', 'hours', 'note'))
    print('-' * 100)
    for j, g, s, e in sorted(rows, key=lambda r: (r[2], r[0].name)):
        print('%-22s %-6s %8.1f %8.1f %7.1f  %s'
              % (j.name, ','.join(map(str, g)) or 'cpu', s, e, e - s, j.note[:40]))
    gpu_h = sum((e - s) * len(g) for _j, g, s, e in rows)
    print('-' * 100)
    left = [n for n, st in status.items() if st == PENDING]
    if left:
        print('never runnable: %s' % ', '.join(left))
    print('estimated wall time  %.1f h  (%.1f days)' % (now, now / 24))
    print('GPU-hours            %.1f on %d GPUs  -> utilisation %.0f%%'
          % (gpu_h, len(gpus), 100 * gpu_h / max(1e-9, now * len(gpus))))
    arms = {}
    for j, _g, _s, e in rows:
        arms[j.arm] = max(arms.get(j.arm, 0.0), e)
    print('arm finishes at      %s' % '  '.join('%s %.0fh' % kv for kv in
                                                sorted(arms.items(), key=lambda kv: kv[1])))
    print('Durations are rough estimates (see jobs.py). Measured ones replace them after '
          'each job finishes.')


# ====================================================================== the runner
class Runner(object):
    def __init__(self, jobs, gpus, cli):
        self.cli = cli
        self.jobs = jobs
        self.names = {j.name for j in jobs}
        self.gpus = list(gpus)
        self.free = list(gpus)
        self.run_dir = cli.run_dir
        self.state_dir = os.path.join(self.run_dir, 'state')
        self.log_dir = os.path.join(self.run_dir, 'logs')
        for d in (self.state_dir, self.log_dir):
            os.makedirs(d, exist_ok=True)
        self.unit = cli.simulate or 3600.0           # seconds per scheduling hour
        self.t0 = time.time()
        self.procs = {}                              # name -> (Popen, log file, gpus)
        self.info = {j.name: {} for j in jobs}
        self.status = {}
        for j in jobs:
            done = os.path.isfile(self.stamp(j.name))
            self.status[j.name] = DONE if done else PENDING
        for j in jobs:
            # a report over everything (analysis) is stale the moment anything re-runs
            if getattr(j, 'always', False) and any(
                    s == PENDING for n, s in self.status.items() if n != j.name):
                self.status[j.name] = PENDING
            if j.gpus > len(self.gpus):
                self.status[j.name] = FAILED
                self.info[j.name]['reason'] = 'needs %d GPUs, %d given' % (j.gpus,
                                                                          len(self.gpus))
        self.main_log = open(os.path.join(self.run_dir, 'run_all.log'), 'a',
                             encoding='utf-8')

    # ---------------------------------------------------------------- bookkeeping
    def stamp(self, name):
        return os.path.join(self.state_dir, '%s.done' % name)

    def now_h(self):
        return (time.time() - self.t0) / self.unit

    def log(self, msg):
        line = '[%s +%6.2fh] %s' % (time.strftime('%Y-%m-%d %H:%M:%S'), self.now_h(), msg)
        print(line, flush=True)
        self.main_log.write(line + '\n')
        self.main_log.flush()

    def log_text(self, name):
        """What the most recent attempt of a job printed."""
        try:
            with open(os.path.join(self.log_dir, '%s.log' % name), encoding='utf-8',
                      errors='replace') as f:
                text = f.read()
        except OSError:
            return ''
        return text.rsplit(HEADER, 1)[-1]

    def write_status(self):
        rows = {}
        for j in self.jobs:
            i = self.info[j.name]
            rows[j.name] = dict(i, status=self.status[j.name], gpus_needed=j.gpus,
                                log=os.path.join(self.log_dir, '%s.log' % j.name))
        tmp = os.path.join(self.run_dir, 'status.json.tmp')
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump({'updated': time.strftime('%Y-%m-%d %H:%M:%S'), 'gpus': self.gpus,
                       'simulate': self.cli.simulate, 'jobs': rows}, f, indent=1)
        os.replace(tmp, os.path.join(self.run_dir, 'status.json'))

    # ---------------------------------------------------------------- lifecycle
    def ready(self):
        out = []
        for j in self.jobs:
            if self.status[j.name] != PENDING:
                continue
            if not all(self.status[d] == DONE for d in j.deps):
                continue
            if not all(self.status[a] in FINISHED for a in j.after if a in self.names):
                continue
            out.append(j)
        return out

    def propagate(self):
        changed = True
        while changed:
            changed = False
            for j in self.jobs:
                if self.status[j.name] != PENDING:
                    continue
                bad = [d for d in j.deps if self.status[d] in NOT_OK]
                if bad:
                    self.status[j.name] = BLOCKED
                    self.info[j.name]['reason'] = '%s %s' % (bad[0], self.status[bad[0]])
                    self.log('blocked %-22s (%s)' % (j.name, self.info[j.name]['reason']))
                    changed = True

    def argv(self, j):
        if self.cli.simulate:
            code = ('import os, sys, time; print("CUDA_VISIBLE_DEVICES=%%r" %% '
                    'os.environ.get("CUDA_VISIBLE_DEVICES")); time.sleep(%f); sys.exit(%d)'
                    % (j.hours * self.cli.simulate,
                       3 if j.name in (self.cli.simulate_fail or ()) else 0))
            return [sys.executable, '-c', code]
        return j.argv(sys.executable)

    def start(self, j, gpus):
        try:
            argv = self.argv(j)
        except Exception as e:                       # noqa: BLE001
            self.status[j.name] = FAILED
            self.info[j.name]['reason'] = 'could not build the command: %s' % e
            self.log('FAILED  %-22s %s' % (j.name, self.info[j.name]['reason']))
            return
        env = os.environ.copy()
        threads = str(self.cli.threads)
        env.update({
            'CUDA_VISIBLE_DEVICES': ','.join(map(str, gpus)),
            'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
            'OMP_NUM_THREADS': threads, 'MKL_NUM_THREADS': threads,
            'TOKENIZERS_PARALLELISM': 'false',
            'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
            'PYTHONUNBUFFERED': '1',
            'PYTHONPATH': os.pathsep.join(p for p in (ROOT, env.get('PYTHONPATH')) if p),
        })
        if not j.online:
            # everything was fetched by setup.prefetch; four processes hitting the Hub at
            # once only adds rate limits and half-written cache entries
            env.setdefault('HF_HUB_OFFLINE', '1')
        logf = open(os.path.join(self.log_dir, '%s.log' % j.name), 'a', encoding='utf-8')
        logf.write('\n%s%s\n' % (HEADER, json.dumps({
            'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'gpus': gpus, 'cwd': j.cwd,
            'argv': argv})))
        logf.flush()
        kw = ({'start_new_session': True} if os.name == 'posix'
              else {'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP})
        try:
            p = subprocess.Popen(argv, cwd=j.cwd, env=env, stdout=logf,
                                 stderr=subprocess.STDOUT, **kw)
        except OSError as e:
            logf.close()
            self.status[j.name] = FAILED
            self.info[j.name]['reason'] = 'could not start: %s' % e
            self.log('FAILED  %-22s %s' % (j.name, self.info[j.name]['reason']))
            return
        for g in gpus:
            self.free.remove(g)
        self.procs[j.name] = (p, logf, gpus)
        self.status[j.name] = RUNNING
        i = self.info[j.name]
        i.update({'gpus': gpus, 'start_h': round(self.now_h(), 3), 'pid': p.pid,
                  'started': time.strftime('%Y-%m-%d %H:%M:%S'),
                  'attempts': i.get('attempts', 0) + 1, 'argv': argv})
        i.pop('reason', None)
        self.log('start   %-22s gpus=%-4s est %.1fh  -> %s'
                 % (j.name, ','.join(map(str, gpus)) or 'cpu', j.hours,
                    os.path.relpath(logf.name, ROOT)))

    def reap(self):
        for name, (p, logf, gpus) in list(self.procs.items()):
            rc = p.poll()
            if rc is None:
                continue
            logf.close()
            del self.procs[name]
            self.free = sorted(self.free + gpus)
            i = self.info[name]
            hours = self.now_h() - i['start_h']
            i.update({'rc': rc, 'hours': round(hours, 3),
                      'ended': time.strftime('%Y-%m-%d %H:%M:%S')})
            if rc == 0:
                self.status[name] = DONE
                with open(self.stamp(name), 'w', encoding='utf-8') as f:
                    json.dump(i, f, indent=1)
                if not self.cli.simulate:
                    d = load_durations(self.run_dir)
                    d[name] = round(hours, 3)
                    with open(os.path.join(self.run_dir, 'durations.json'), 'w',
                              encoding='utf-8') as f:
                        json.dump(d, f, indent=1)
                self.log('done    %-22s %.2fh' % (name, hours))
            else:
                self.status[name] = FAILED
                tail = [l.rstrip() for l in self.log_text(name).splitlines() if l.strip()]
                i['reason'] = 'exit %d: %s' % (rc, tail[-1][:160] if tail else '')
                self.log('FAILED  %-22s exit %d after %.2fh' % (name, rc, hours))
                for line in tail[-6:]:
                    self.log('        | %s' % line[:200])

    def running_view(self):
        out = []
        for name, (_p, _f, gpus) in self.procs.items():
            j = next(x for x in self.jobs if x.name == name)
            end = self.info[name]['start_h'] + j.hours
            out.append((j, gpus, max(end, self.now_h() + 0.1)))
        return out

    def stop(self):
        if not self.procs:
            return
        self.log('stopping %d running job(s)' % len(self.procs))
        for p, _f, _g in self.procs.values():
            try:
                if os.name == 'posix':
                    os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                else:
                    p.terminate()
            except (OSError, ProcessLookupError):
                pass
        deadline = time.time() + 30
        for name, (p, logf, _g) in list(self.procs.items()):
            try:
                p.wait(timeout=max(0.1, deadline - time.time()))
            except subprocess.TimeoutExpired:
                try:
                    if os.name == 'posix':
                        os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                    else:
                        p.kill()
                except (OSError, ProcessLookupError):
                    pass
            logf.close()
            self.status[name] = INTERRUPTED
            self.info[name]['reason'] = 'interrupted; re-run to resume'
        self.procs.clear()

    # ---------------------------------------------------------------- main loop
    def run(self):
        todo = [j.name for j in self.jobs if self.status[j.name] == PENDING]
        self.log('run_all: %d jobs selected, %d already done, gpus %s, run dir %s%s'
                 % (len(self.jobs), len(self.jobs) - len(todo),
                    ','.join(map(str, self.gpus)), self.run_dir,
                    '  [SIMULATION %.3gs/h]' % self.cli.simulate if self.cli.simulate else ''))
        last_beat = time.time()
        try:
            while True:
                self.reap()
                self.propagate()
                ready = []
                for j in self.ready():
                    reason = j.gate(self) if (j.gate and not self.cli.force_gates
                                              and not self.cli.simulate) else None
                    if reason:
                        self.status[j.name] = GATED
                        self.info[j.name]['reason'] = reason
                        self.log('gated   %-22s %s' % (j.name, reason))
                        continue
                    ready.append(j)
                self.propagate()
                starts = plan_starts(self.now_h(), ready, self.free, self.running_view(),
                                     self.cli.max_cpu_jobs)
                for j, gpus in starts:
                    self.start(j, gpus)
                self.write_status()
                if not self.procs and not starts:
                    stuck = [n for n, s in self.status.items() if s == PENDING]
                    for n in stuck:
                        self.status[n] = BLOCKED
                        self.info[n]['reason'] = 'dependencies can never succeed'
                    break
                if self.cli.heartbeat and time.time() - last_beat > 60 * self.cli.heartbeat:
                    last_beat = time.time()
                    self.log('running: %s   free gpus: %s' % (
                        ', '.join('%s[%s] %.1fh' % (n, ','.join(map(str, g)),
                                                    self.now_h() - self.info[n]['start_h'])
                                  for n, (_p, _f, g) in self.procs.items()) or '-',
                        ','.join(map(str, self.free)) or '-'))
                time.sleep(self.cli.poll)
        except KeyboardInterrupt:
            self.stop()
            self.write_status()
            self.report()
            raise SystemExit(130)
        self.write_status()
        return self.report()

    def report(self):
        print()
        print('%-22s %-12s %-6s %7s  %s' % ('job', 'status', 'gpus', 'hours', 'reason'))
        print('-' * 100)
        for j in self.jobs:
            i = self.info[j.name]
            print('%-22s %-12s %-6s %7s  %s'
                  % (j.name, self.status[j.name],
                     ','.join(map(str, i.get('gpus') or [])) or ('cpu' if not j.gpus else '-'),
                     fmt_h(i.get('hours')), (i.get('reason') or '')[:60]))
        bad = [n for n, s in self.status.items() if s != DONE]
        print('-' * 100)
        print('%d/%d done%s' % (len(self.jobs) - len(bad), len(self.jobs),
                                ('; not done: ' + ', '.join(bad)) if bad else ''))
        return 0 if not bad else 1


# ====================================================================== CLI
def print_status(run_dir):
    path = os.path.join(run_dir, 'status.json')
    if not os.path.isfile(path):
        raise SystemExit('no status at %s' % path)
    with open(path, encoding='utf-8') as f:
        st = json.load(f)
    print('updated %s   gpus %s%s' % (st['updated'], st['gpus'],
                                      '   (simulation)' if st.get('simulate') else ''))
    print('%-22s %-12s %-6s %8s %8s  %s' % ('job', 'status', 'gpus', 'start h', 'hours',
                                            'reason'))
    for name, r in st['jobs'].items():
        print('%-22s %-12s %-6s %8s %8s  %s'
              % (name, r['status'], ','.join(map(str, r.get('gpus') or [])) or '-',
                 fmt_h(r.get('start_h')), fmt_h(r.get('hours')),
                 (r.get('reason') or '')[:60]))


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--gpus', default=None,
                   help='comma-separated GPU indices (default: every GPU nvidia-smi lists)')
    p.add_argument('--only', nargs='+', default=None,
                   help='arms (ppdpp epo sr tom om rt analysis) or job names')
    p.add_argument('--dry_run', action='store_true', help='print the estimated schedule')
    p.add_argument('--list', action='store_true', help='print the jobs and exit')
    p.add_argument('--status', action='store_true', help='print the last status and exit')
    p.add_argument('--reset', nargs='+', default=None,
                   help='delete done-stamps for these jobs (or "all") so they re-run')
    p.add_argument('--run_dir', default=None)
    p.add_argument('--max_cpu_jobs', type=int, default=2)
    p.add_argument('--threads', type=int, default=None,
                   help='OMP/MKL threads per job (default: CPUs / GPUs)')
    p.add_argument('--force_gates', action='store_true',
                   help='start gated jobs anyway (e.g. Omega corpus after a STOP probe)')
    p.add_argument('--poll', type=float, default=5.0)
    p.add_argument('--heartbeat', type=float, default=30.0, help='minutes; 0 = off')
    p.add_argument('--simulate', type=float, default=None, metavar='SECONDS_PER_HOUR',
                   help='test the scheduler: stand-in jobs sleep their estimate x this')
    p.add_argument('--simulate_fail', nargs='+', default=None,
                   help='with --simulate, make these jobs exit non-zero')
    cli = p.parse_args()

    cli.run_dir = os.path.abspath(cli.run_dir or os.path.join(
        ROOT, 'runs-simulate' if cli.simulate else 'runs'))
    if cli.status:
        return print_status(cli.run_dir)

    all_jobs = J.build(ROOT)
    validate(all_jobs)
    jobs = select(all_jobs, cli.only)
    measured = {} if cli.simulate else load_durations(cli.run_dir)
    for j in jobs:
        if isinstance(measured.get(j.name), (int, float)):
            j.hours = measured[j.name]
    assign_tails(jobs)

    if cli.reset:
        names = [j.name for j in all_jobs] if 'all' in cli.reset else cli.reset
        for n in names:
            path = os.path.join(cli.run_dir, 'state', '%s.done' % n)
            if os.path.isfile(path):
                os.remove(path)
                print('reset %s' % n)
        return 0

    if cli.list:
        print('%-22s %-5s %6s %6s %5s  %-40s %s' % ('job', 'gpus', 'hours', 'chain',
                                                    'prio', 'deps', 'note'))
        for j in sorted(jobs, key=_rank):
            print('%-22s %-5s %6.1f %6.1f %5d  %-40s %s'
                  % (j.name, j.gpus or 'cpu', j.hours, j.tail, j.priority,
                     ','.join(j.deps)[:40], j.note))
        return 0

    found = detect_gpus()
    if cli.gpus:
        gpus = [int(x) for x in cli.gpus.split(',') if x.strip()]
    elif found and not cli.simulate:
        gpus = [g[0] for g in found]
    else:
        gpus = [0, 1, 2, 3]
    cli.threads = cli.threads or max(1, (os.cpu_count() or 4) // max(1, len(gpus)))

    if cli.dry_run:
        done = {j.name for j in jobs
                if os.path.isfile(os.path.join(cli.run_dir, 'state', '%s.done' % j.name))}
        print('dry run on gpus %s (%d CPU jobs at once)%s\n'
              % (gpus, cli.max_cpu_jobs, '; %d jobs already done' % len(done) if done else ''))
        return dry_run(jobs, gpus, cli.max_cpu_jobs, done)

    if not cli.simulate:
        if not found:
            raise SystemExit('nvidia-smi lists no GPUs. Use --dry_run to see the plan, or '
                             '--simulate to test the scheduler without GPUs.')
        known = {g[0]: g for g in found}
        missing = [g for g in gpus if g not in known]
        if missing:
            raise SystemExit('GPU(s) %s not present; nvidia-smi lists %s'
                             % (missing, sorted(known)))
        for g in gpus:
            if known[g][2] < 22000:
                print('WARNING gpu %d (%s) has %d MiB; the jobs are sized for 24 GB'
                      % (g, known[g][1], known[g][2]))

    runner = Runner(jobs, gpus, cli)
    if os.name == 'posix':
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt))
    return runner.run()


if __name__ == '__main__':
    sys.exit(main())
