# ---------------------------------------------------------------------------
# VENDORED from csa_core/data_csa.py. Do not edit this copy.
#
# roundtable/ is deliberately standalone -- it imports nothing from the rest of the
# repo, so the folder can be lifted out on its own. The price is that this file can
# drift from the shared one and silently make this arm's numbers incomparable, so
# selftest.py compares it against csa_core whenever csa_core is reachable.
#
# To change the scoring contract: edit csa_core/data_csa.py, then re-run
#     python roundtable/vendor.py
# ---------------------------------------------------------------------------
"""Load the CSA scenarios and rebuild the train/valid/test split from scratch.

The split MUST come out identical for every baseline, or the
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

The scenarios come from the combined file paths.SCENARIOS_FILENAME when it can be found,
and from the per-domain files under data/raw/ otherwise. Both hold the same uids, so the
split is identical either way; see paths.py for the seven scenarios whose content differs.
"""
import collections
import json
import os
import random

import _paths as paths

SEED = 0
TRAIN_FRAC = 0.70
VALID_FRAC = 0.10

_CACHE = {}


def _unwrap_schema(row):
    """Flatten `settlement_schema: {settlement: {decisions: ...}}` to `{decisions: ...}`.

    Two of the 1100 scenarios (family_friends_informal::scenario_18 and
    manufacturing::scenario_7) nest the schema one level deeper under a `settlement` key,
    with an optional `instructions` string beside it. The contents are otherwise
    identical, so this is a shape variant rather than corruption -- normalising it here
    keeps both scenarios in the benchmark instead of dropping them for a formatting
    difference. In-memory only; the files on disk are untouched.
    """
    sch = row.get('settlement_schema')
    if isinstance(sch, dict) and 'decisions' not in sch:
        inner = sch.get('settlement')
        if isinstance(inner, dict) and 'decisions' in inner:
            merged = dict(inner)
            if 'instructions' in sch:
                merged['instructions'] = sch['instructions']
            row['settlement_schema'] = merged
    return row


def _num(scenario_id):
    """Sort key for scenario_7 vs scenario_70. Lexical order would interleave them."""
    tail = str(scenario_id).rsplit('_', 1)[-1]
    return int(tail) if tail.isdigit() else 0


def source(raw_dir=None):
    """Where load_raw() reads from: the combined file, or the per-domain directory."""
    if raw_dir:
        return raw_dir
    return paths.find_scenarios_file() or paths.find_raw()


def _read_combined(path):
    """{filename-stem domain: [rows]} from the single-file scenario set."""
    with open(path, encoding='utf-8') as f:
        got = json.load(f)
    by_domain = collections.defaultdict(list)
    for r in got:
        by_domain[paths.DOMAIN_ALIASES.get(r['domain'], r['domain'])].append(r)
    return by_domain


def load_raw(raw_dir=None, per_domain=None):
    """Every configured scenario, with the composite uid attached.

    An explicit raw_dir reads the per-domain files; otherwise the combined file is used
    when paths.find_scenarios_file() finds one. Two normalisations happen here, both
    load-time only -- the files on disk are never rewritten:

      * `domain` is forced to the FILENAME STEM. family_friends_informal_scenarios.json
        carries an internal domain of `friends_family_informal`, transposed, and letting
        that through would make the uid unpredictable from the domain list.
      * each domain is capped at paths.SCENARIOS_PER_DOMAIN, taking the LOWEST
        scenario_ids. Head rather than sample, so raising the cap only ever adds
        scenarios and never reshuffles the ones already in use.
    """
    cap = paths.SCENARIOS_PER_DOMAIN if per_domain is None else per_domain
    combined = None if raw_dir else paths.find_scenarios_file()
    by_domain = _read_combined(combined) if combined else None
    if not combined:
        raw_dir = raw_dir or paths.find_raw()

    rows = []
    for domain in paths.DOMAINS:
        if combined:
            got = list(by_domain.get(domain, ()))
            if len(got) < (cap or 1):
                # A silent short domain would change the benchmark, not just its size.
                raise SystemExit(
                    '%s holds %d scenarios for %s, but %d are configured. Use the '
                    'per-domain files instead: export CSA_RAW_DIR=/path/to/raw'
                    % (combined, len(got), domain, cap))
        else:
            with open(os.path.join(raw_dir, '%s_scenarios.json' % domain),
                      encoding='utf-8') as f:
                got = json.load(f)
        got.sort(key=lambda r: _num(r['scenario_id']))
        if cap:
            got = got[:cap]
        for r in got:
            r['domain'] = domain
            r['uid'] = '%s::%s' % (domain, r['scenario_id'])
            _unwrap_schema(r)
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
    print('scenarios: %s' % source())
    print({k: len(v) for k, v in d.items()})
    bad = check_invariants(load_raw())
    print('invariant violations: %d' % len(bad))
    for b in bad[:5]:
        print('  ' + b)
