"""Load the CSA scenarios and rebuild the train/valid/test split from scratch.

The split MUST come out identical to the one the other baselines used (99/9/42), or the
three arms are not comparable. Rather than read their split files, the same deterministic
procedure is reimplemented here and selftest.py checks the resulting uid lists against
the published ones. Re-deriving beats reading: if the procedure ever drifts, the check
fails loudly instead of the numbers quietly diverging.

The procedure, which must not be "improved":
  * uid = "<domain>::<scenario_id>". scenario_id is only unique WITHIN a domain -- all
    three files number scenario_1..50 -- so without the composite key the splits collide.
  * bucket by (domain, num_agents), iterate buckets in sorted key order
  * ONE random.Random(0) shared across every bucket, so RNG state carries between them
  * within a bucket, sort by uid, shuffle, then take 70% / max(1, 10%) / remainder
"""
import collections
import json
import os
import random

import paths

SEED = 0
TRAIN_FRAC = 0.70
VALID_FRAC = 0.10

_CACHE = {}


def load_raw(raw_dir=None):
    """All 150 scenarios, with the composite uid attached."""
    raw_dir = raw_dir or paths.find_raw()
    rows = []
    for domain in paths.DOMAINS:
        with open(os.path.join(raw_dir, '%s_scenarios.json' % domain),
                  encoding='utf-8') as f:
            for r in json.load(f):
                r['uid'] = '%s::%s' % (r['domain'], r['scenario_id'])
                rows.append(r)
    return rows


def split_rows(rows, seed=SEED, train_frac=TRAIN_FRAC, valid_frac=VALID_FRAC):
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
    return {'train': train, 'valid': valid, 'test': test}


def load(split=None, raw_dir=None):
    key = raw_dir or '_default'
    if key not in _CACHE:
        d = split_rows(load_raw(raw_dir))
        seen = set()
        for name in ('train', 'valid', 'test'):
            uids = {r['uid'] for r in d[name]}
            if seen & uids:
                raise SystemExit('scenario leaked across splits: %s'
                                 % sorted(seen & uids)[:5])
            seen |= uids
        _CACHE[key] = d
    d = _CACHE[key]
    return d[split] if split else d


def case_index(raw_dir=None):
    return {r['uid']: r for r in load_raw(raw_dir)}


def check_invariants(rows):
    """The assumptions the environment and the reward rely on. Returns a list of
    violations; empty means clean."""
    bad = []
    for r in rows:
        ids = {a['agent_id'] for a in r['agents']}
        order = r['interaction_config']['turn_order']
        dm = r['decision_maker']
        if set(order) - ids:
            bad.append('%s: turn_order names unknown agents' % r['uid'])
        if dm not in ids:
            bad.append('%s: decision_maker not among agents' % r['uid'])
        if dm not in order:
            bad.append('%s: decision_maker never speaks' % r['uid'])
        for fid, fact in r['private_facts'].items():
            if fact['owner'] == dm:
                bad.append('%s: %s owned by the chair' % (r['uid'], fid))
            if fid in r['views'].get(dm, []):
                bad.append('%s: %s visible to the chair' % (r['uid'], fid))
        if 'decisions' not in r['settlement_schema']:
            bad.append('%s: settlement_schema has no decisions block' % r['uid'])
    return bad


if __name__ == '__main__':
    d = load()
    print('raw dir: %s' % paths.find_raw())
    print({k: len(v) for k, v in d.items()})
    bad = check_invariants(load_raw())
    print('invariant violations: %d' % len(bad))
    for b in bad[:5]:
        print('  ' + b)
