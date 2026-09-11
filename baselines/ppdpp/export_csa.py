"""Export Coevolving Social Agents into the line-per-dict format utils.load_dataset expects.

The Hub ships one JSON per domain, not a train/test split, so PPDPP needs the split
serialised into the format its loader eval()s. The split itself comes from
csa_core.data_csa -- this script does not recompute it, because a second copy of the
procedure is how PPDPP would silently end up on a different benchmark from the other
arms.

    python export_csa.py --out_dir ./data
"""
import argparse
import collections
import io
import json
import os
import random
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from csa_core import data_csa, paths            # noqa: E402


def load_raw(raw_dir):
    rows = []
    for domain in DOMAINS:
        with open(os.path.join(raw_dir, '%s_scenarios.json' % domain), encoding='utf-8') as f:
            for r in json.load(f):
                r['uid'] = '%s::%s' % (r['domain'], r['scenario_id'])
                rows.append(r)
    return rows


def check_invariants(rows):
    """Fail loudly on the assumptions the environment and verifier rely on."""
    problems = []
    for r in rows:
        ids = {a['agent_id'] for a in r['agents']}
        order = r['interaction_config']['turn_order']
        dm = r['decision_maker']
        if set(order) - ids:
            problems.append('%s: turn_order names unknown agents' % r['uid'])
        if dm not in ids:
            problems.append('%s: decision_maker not among agents' % r['uid'])
        if dm not in order:
            problems.append('%s: decision_maker never speaks' % r['uid'])
        for fid, fact in r['private_facts'].items():
            if fact['owner'] == dm:
                problems.append('%s: %s owned by the chair' % (r['uid'], fid))
            if fid in r['views'].get(dm, []):
                problems.append('%s: %s visible to the chair' % (r['uid'], fid))
        if 'decisions' not in r['settlement_schema']:
            problems.append('%s: settlement_schema has no decisions block' % r['uid'])
    return problems


def audit_checks(rows):
    """Checks that cannot ever pass, because they read a field the schema never defines.

    Reported rather than fatal: one such defect exists in the shipped corpus, and it
    must be known when interpreting results rather than silently depressing them.
    """
    import re
    dead = []
    for r in rows:
        schema = set((r['settlement_schema'].get('decisions') or {}).keys())
        for cid, expr in r['content_checks'].items():
            try:
                compile(expr, '<check>', 'eval')
            except SyntaxError:
                dead.append((r['uid'], cid, 'does not parse'))
                continue
            fields = re.findall(r"decisions\[['\"]([^'\"]+)['\"]\]", expr)
            fields += re.findall(r"decisions\.get\(['\"]([^'\"]+)['\"]", expr)
            for fld in fields:
                if fld not in schema:
                    dead.append((r['uid'], cid, "reads decisions[%r], absent from schema" % fld))
    return dead


def split(rows, seed, train_frac, valid_frac):
    buckets = collections.defaultdict(list)
    for r in rows:
        buckets[(r['domain'], r['num_agents'])].append(r)
    rng = random.Random(seed)
    train, valid, test = [], [], []
    for key in sorted(buckets):
        b = sorted(buckets[key], key=lambda r: r['uid'])
        rng.shuffle(b)
        n = len(b)
        n_tr, n_va = int(train_frac * n), max(1, int(valid_frac * n))
        train += b[:n_tr]
        valid += b[n_tr:n_tr + n_va]
        test += b[n_tr + n_va:]
    return train, valid, test


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', default='./data')
    p.add_argument('--raw_dir', default=None,
                   help='override the raw scenario directory; normally discovered')
    args = p.parse_args()

    # The split is NOT recomputed here. csa_core.data_csa owns it, and every other arm
    # reads it from there -- a second copy of the procedure in this file is exactly how
    # PPDPP would end up evaluating a different benchmark from everyone else. This script
    # only serialises that split into the line-per-dict format load_dataset() eval()s.
    rows = data_csa.load_raw(args.raw_dir)
    problems = data_csa.check_invariants(rows)
    if problems:
        raise SystemExit('dataset invariants violated:\n  '
                         + '\n  '.join(problems))
    print('invariants: all %d scenarios pass (%d domains x %d)'
          % (len(rows), len(paths.DOMAINS), paths.SCENARIOS_PER_DOMAIN))

    dead = audit_checks(rows)
    print('unsatisfiable content checks: %d' % len(dead))
    for uid, cid, why in dead:
        print('  WARNING %s %s %s' % (uid, cid, why))

    parts = [(k, data_csa.load(k, args.raw_dir)) for k in ('train', 'valid', 'test')]
    seen = set()
    for name, part in parts:
        overlap = seen & {r['uid'] for r in part}
        if overlap:
            raise SystemExit('scenario leaked across splits: %s' % sorted(overlap)[:5])
        seen |= {r['uid'] for r in part}

    os.makedirs(args.out_dir, exist_ok=True)
    for name, part in parts:
        path = os.path.join(args.out_dir, 'csa-%s.txt' % name)
        with open(path, 'w', encoding='utf-8') as f:
            for r in part:
                # load_dataset calls eval() on each line
                f.write(repr(r) + '\n')
        print('wrote %-22s %4d scenarios' % (path, len(part)))


if __name__ == '__main__':
    main()
