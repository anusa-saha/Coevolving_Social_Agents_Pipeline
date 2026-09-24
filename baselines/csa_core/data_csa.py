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

EXPLICIT SPLITS OVERRIDE ALL OF THAT. When data/splits/ is present (see
paths.find_splits_dir) the split is READ, not derived: train.json is train, test_all.json
is test, and the two eval_group files are available as the extra splits `test_seen` and
`test_unseen`. Nothing is shuffled and nothing is capped -- the files decide, so every
arm evaluates on exactly the scenarios the benchmark prescribes.

Two consequences of reading rather than deriving, both deliberate:

  * there is no validation file, so `valid` is carved out of the TRAIN rows by the same
    deterministic bucket procedure described above (10% per (domain, num_agents) bucket,
    random.Random(0)). Test is never touched by it.
  * 155 scenarios appear in BOTH train.json and test_seen.json, byte-identical. That is
    train/test overlap, and the seen-domain numbers are optimistic because of it. The
    files are authoritative, so the loader keeps them and says so loudly once per process
    instead of failing; test_unseen is the clean generalisation measurement.
"""
import collections
import json
import os
import random

from csa_core import paths

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
    """Where the scenarios read from: the split directory, the combined file, or the
    per-domain directory."""
    if raw_dir:
        return raw_dir
    return (paths.find_splits_dir() or paths.find_scenarios_file()
            or paths.find_raw())


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
    if per_domain is None and is_explicit(raw_dir):
        # The split files ARE the scenario set; there is no separate corpus to re-read.
        # Deduplicated by uid, because the 155 overlapping scenarios are one scenario
        # each -- case_index and the invariant checks want every scenario exactly once.
        d = load(raw_dir=raw_dir)
        rows, seen = [], set()
        for name in ('train', 'valid', 'test') + EVAL_GROUPS:
            for r in d[name]:
                if r['uid'] not in seen:
                    seen.add(r['uid'])
                    rows.append(r)
        return rows

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


# ------------------------------------------------------------- explicit splits
# Extra splits beyond train/valid/test. They are views on the evaluation set rather than
# a further partition of it: test == test_seen + test_unseen.
EVAL_GROUPS = ('test_seen', 'test_unseen')

_WARNED = set()


def _prepare(row, seen_uids):
    """Attach the uid and normalise, exactly as load_raw does for the derived split."""
    row['domain'] = paths.DOMAIN_ALIASES.get(row['domain'], row['domain'])
    row['uid'] = '%s::%s' % (row['domain'], row['scenario_id'])
    _unwrap_schema(row)
    if row['uid'] in seen_uids:
        # Within ONE file a repeated uid is corruption, not a documented overlap: the two
        # rows would be indistinguishable to case_index and one would silently win.
        raise SystemExit('duplicate uid %s within a single split file' % row['uid'])
    seen_uids.add(row['uid'])
    return row


def _read_split_file(path):
    with open(path, encoding='utf-8') as f:
        got = json.load(f)
    seen = set()
    return [_prepare(r, seen) for r in got]


def _carve_valid(rows, seed=SEED, valid_frac=VALID_FRAC):
    """Hold a validation slice out of the train rows.

    The split files ship no validation set, but SFT early-stopping and `--split valid`
    both need one. Rather than invent a second procedure, this reuses the one in
    split_rows: bucket by (domain, num_agents), walk buckets in sorted key order with ONE
    random.Random(seed) so RNG state carries between them, sort each bucket by uid and
    shuffle. The tail of each bucket becomes valid, so every bucket contributes and the
    result is identical on every machine. Train order from the file is not preserved --
    the split has to be reproducible, and file order is not a property anything relies on.
    """
    buckets = collections.defaultdict(list)
    for r in rows:
        buckets[(r['domain'], r['num_agents'])].append(r)
    rng = random.Random(seed)
    train, valid = [], []
    for key in sorted(buckets):
        b = sorted(buckets[key], key=lambda r: r['uid'])
        rng.shuffle(b)
        n_va = max(1, int(valid_frac * len(b)))
        train += b[:len(b) - n_va]
        valid += b[len(b) - n_va:]
    return train, valid


def _warn_once(key, msg):
    if key not in _WARNED:
        _WARNED.add(key)
        print(msg)


def load_explicit(splits_dir):
    """The prescribed split, read verbatim from splits_dir.

    Returns train/valid/test plus the two eval_group views. `valid` comes out of train;
    the test files are passed through untouched.
    """
    out = {}
    for name in ('train',) + EVAL_GROUPS:
        out[name] = _read_split_file(os.path.join(splits_dir, paths.SPLIT_FILES[name]))
    out['test'] = _read_split_file(os.path.join(splits_dir, paths.SPLIT_FILES['test']))

    group = {r['uid'] for r in out['test_seen']} | {r['uid'] for r in out['test_unseen']}
    if {r['uid'] for r in out['test']} - group:
        raise SystemExit('%s holds scenarios that are in neither eval_group file'
                         % paths.SPLIT_FILES['test'])

    out['train'], out['valid'] = _carve_valid(out['train'])

    # The documented overlap. Reported, not repaired: the files are the benchmark, and a
    # loader that quietly dropped 155 scenarios would make the arms incomparable to
    # anyone else running the same files.
    tr = {r['uid'] for r in out['train']} | {r['uid'] for r in out['valid']}
    leaked = sorted(tr & {r['uid'] for r in out['test']})
    if leaked:
        _warn_once('leak', (
            '[data_csa] WARNING: %d of the %d test scenarios also appear in '
            'train.json, byte-identical.\n'
            '           Seen-domain test numbers are therefore optimistic; '
            'test_unseen (%d scenarios,\n'
            '           two domains absent from training) is the clean '
            'generalisation measurement.\n'
            '           First five: %s'
            % (len(leaked), len(out['test']), len(out['test_unseen']),
               leaked[:5])))
    return out


def is_explicit(raw_dir=None):
    """True when the split is being read from files rather than derived."""
    return raw_dir is None and paths.find_splits_dir() is not None


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
    """The split. 'train', 'valid', 'test', or -- with explicit splits -- 'test_seen'
    and 'test_unseen'. No argument returns the whole dict."""
    key = raw_dir or '_default'
    if key not in _CACHE:
        if is_explicit(raw_dir):
            d = load_explicit(paths.find_splits_dir())
        else:
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
    if split and split not in d:
        raise SystemExit('unknown split %r; this configuration has %s'
                         % (split, ', '.join(sorted(d))))
    return d[split] if split else d


def expected_total():
    """How many DISTINCT scenarios the configured benchmark holds.

    Not sum(len(split)): with explicit splits the overlapping 155 are counted once here
    and twice there.
    """
    if is_explicit():
        return len(load_raw())
    return len(paths.DOMAINS) * paths.SCENARIOS_PER_DOMAIN


def audit_splits(d=None):
    """Check the properties that must hold under ANY configuration, and describe it.

    Returns a one-line summary for selftests to print; raises AssertionError on a real
    problem. The train/test overlap that ships with the explicit split files is reported,
    not failed -- load_explicit already warns about it.
    """
    d = d if d is not None else load()
    got = {k: len(v) for k, v in d.items()}
    total = expected_total()

    if not is_explicit():
        assert sum(got.values()) == total, (got, total)
        uids = [r['uid'] for k in ('train', 'valid', 'test') for r in d[k]]
        assert len(set(uids)) == total, 'split overlaps or drops scenarios'
        return ('%d domains x %d = %d -> %d/%d/%d'
                % (len(paths.DOMAINS), paths.SCENARIOS_PER_DOMAIN, total,
                   got['train'], got['valid'], got['test']))

    tr = {r['uid'] for r in d['train']}
    va = {r['uid'] for r in d['valid']}
    assert not (tr & va), 'valid was carved out of train but they overlap'
    seen = {r['uid'] for r in d['test_seen']}
    unseen = {r['uid'] for r in d['test_unseen']}
    assert not (seen & unseen), 'test_seen and test_unseen overlap'
    assert {r['uid'] for r in d['test']} <= (seen | unseen),         'test holds scenarios in neither eval_group'
    assert len(tr | va | seen | unseen) == total, 'scenarios lost between splits'
    return ('explicit split from %s: %d/%d train/valid, %d test (%d seen, %d unseen), '
            '%d distinct scenarios, %d shared by train and test'
            % (source(), got['train'], got['valid'], got['test'],
               got['test_seen'], got['test_unseen'], total,
               len((tr | va) & {r['uid'] for r in d['test']})))


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
    print(audit_splits(d))
    print('domains: %s' % ', '.join(
        '%s=%d' % kv for kv in sorted(collections.Counter(
            r['domain'] for r in load_raw()).items())))
    bad = check_invariants(load_raw())
    print('invariant violations: %d' % len(bad))
    for b in bad[:5]:
        print('  ' + b)
