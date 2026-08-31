"""Deterministic scoring of a settlement against the dataset's executable checks.

Written from scratch for this package, but the semantics must match what the other
baselines report or the three arms cannot be compared. selftest.py recomputes cbar, pbar
and disclosure for 297 published episodes and asserts agreement with the stored values,
which validates this file against the reference implementation without importing it.

No model is involved. Checks are Python expressions shipped with the dataset, e.g.

    decisions['medication_approved'] in ['dalbavancin', 'Dalbavancin']
    'PF1' in credited_facts and 'PF1' in justification_fact_ids and 'PF1' in revealed

evaluated over the extracted settlement plus the facts observed to have been disclosed.
"""
import re
import unicodedata

from detectors_sr import REVEAL_THRESHOLD, overlap

# ---------------------------------------------------------------- canonicalisation
_THOUSANDS = re.compile(r'(?<=\d),(?=\d)')
_WS = re.compile(r'\s+')
_UNITS = [
    (re.compile(r'\bdegrees?\s+c(elsius)?\b'), 'c'),
    (re.compile(r'\bdegrees?\s+f(ahrenheit)?\b'), 'f'),
    (re.compile(r'\bhours?\b'), 'h'),
    (re.compile(r'\bmilligrams?\b'), 'mg'),
    (re.compile(r'\bpercent\b'), '%'),
    # "3:00 PM" and "3 PM" are the same deadline; several checks compare times by exact
    # string equality, which otherwise scores formatting rather than content.
    (re.compile(r'\b(\d{1,2}):00\b'), r'\1'),
    (re.compile(r'\b([ap])\.m\.'), r'\1m'),
]


def canonical(s):
    """Applied to BOTH sides of a comparison, so formatting differences do not read as
    reasoning failures."""
    if not isinstance(s, str):
        return s
    s = unicodedata.normalize('NFKD', s)
    for a, b in (('–', '-'), ('—', '-'), ('−', '-'),
                 ('°', ' degrees '), ('’', "'")):
        s = s.replace(a, b)
    s = s.lower().strip()
    s = _THOUSANDS.sub('', s)
    for pat, rep in _UNITS:
        s = pat.sub(rep, s)
    return _WS.sub(' ', s).strip(' .;:')


class Missing(str):
    """Stands in for an absent settlement field. Behaves as '' so `==` and `in` return
    False instead of raising -- a missing field must FAIL its check, not crash it."""

    def __new__(cls):
        return super().__new__(cls, '')


class NormStr(str):
    """A settlement value that compares canonically rather than byte-for-byte."""

    def __eq__(self, other):
        if isinstance(other, str):
            return canonical(self) == canonical(other)
        return str.__eq__(self, other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(canonical(self))

    def __contains__(self, item):
        if isinstance(item, str):
            return canonical(item) in canonical(self)
        return str.__contains__(self, item)


class SafeDict(dict):
    """Missing keys yield Missing() through both [] and .get()."""

    def __init__(self, data=None, norm=False):
        super().__init__(data or {})
        self._norm = norm

    def _wrap(self, v):
        return NormStr(v) if self._norm and isinstance(v, str) else v

    def __missing__(self, k):
        return Missing()

    def __getitem__(self, k):
        return self._wrap(super().__getitem__(k)) if k in self else Missing()

    def get(self, k, default=None):
        if k in self:
            return self._wrap(super().get(k))
        return Missing() if default is None else default


class NormList(list):
    """Canonical membership, for `x in credited_facts`. Nested dicts are wrapped too, so
    checks reaching into commitments compare the same way and a missing key fails rather
    than raises."""

    def __init__(self, items=()):
        super().__init__(SafeDict(i, norm=True) if isinstance(i, dict) else i
                         for i in items)

    def __contains__(self, item):
        if isinstance(item, str):
            return any(canonical(item) == canonical(v) for v in self if isinstance(v, str))
        return list.__contains__(self, item)


SAFE_BUILTINS = {'any': any, 'all': all, 'len': len, 'sum': sum, 'str': str, 'int': int,
                 'float': float, 'sorted': sorted, 'set': set, 'abs': abs, 'min': min,
                 'max': max, 'bool': bool, 'round': round,
                 'True': True, 'False': False, 'None': None}


def eval_checks(checks, ctx):
    """{'C1': '<python expr>'} -> {'C1': True/False}.

    Never evaluate these outside the restricted environment: they ship with a downloaded
    dataset and run inside the training loop.
    """
    env = {'__builtins__': SAFE_BUILTINS}
    env.update(ctx)
    out = {}
    for cid, expr in (checks or {}).items():
        try:
            out[cid] = bool(eval(expr, env))
        except Exception:                            # noqa: BLE001
            out[cid] = False
    return out


def build_context(settlement, revealed, norm=False):
    listcls = NormList if norm else list
    s = settlement or {}
    return {
        'decisions': SafeDict(s.get('decisions') or {}, norm=norm),
        'commitments': listcls(s.get('commitments') or []),
        'credited_facts': listcls(s.get('credited_facts') or []),
        'justification_fact_ids': listcls(s.get('justification_fact_ids') or []),
        'revealed': listcls(sorted(revealed)),
    }


def flipped_checks(case):
    """Every check that some decisive fact controls. The denominator for dca."""
    return sorted({c for d in (case.get('decisive_facts') or [])
                   for c in (d.get('flips') or [])})


def resolve_provenance(case, settlement, revealed, threshold=REVEAL_THRESHOLD):
    """Recover the fact ids a settlement is grounded in from its prose, not its labels.

    The chair never sees the string 'PF1' -- private fact ids exist only in the dataset
    and in the advisors' views -- so a provenance check can only be satisfied by guessing.
    Naming the ids in the chair's prompt would hand it the label for information it is
    supposed to elicit. Instead the settlement's own wording is matched back to fact texts
    with the same detector that populates `revealed`.

    Only already-disclosed facts are eligible, so this can never manufacture provenance
    for information that was never pooled. Applied by every baseline; skipping it here
    would give this arm a harsher provenance rule than the others.
    """
    if not isinstance(settlement, dict):
        return settlement
    prose = [str(v) for v in (settlement.get('decisions') or {}).values()]
    for c in settlement.get('commitments') or []:
        if isinstance(c, dict):
            prose.extend(str(c.get(k, '')) for k in ('type', 'target', 'detail'))
    blob = ' '.join(prose)
    hits = {fid for fid in revealed
            if fid in case['private_facts']
            and overlap(case['private_facts'][fid]['text'], blob) >= threshold}
    if not hits:
        return settlement
    out = dict(settlement)
    for field in ('justification_fact_ids', 'credited_facts'):
        have = [x for x in (out.get(field) or []) if isinstance(x, str)]
        out[field] = have + sorted(hits - set(have))
    return out


def schema_valid(case, settlement):
    if not isinstance(settlement, dict):
        return False
    want = case['settlement_schema'].get('decisions')
    if not isinstance(want, dict):
        return 'decisions' in settlement
    got = settlement.get('decisions')
    return isinstance(got, dict) and set(want) <= set(got)


def hallucinated_credit(case, settlement, revealed):
    """Cited fact ids that were never revealed, or that do not exist."""
    if not isinstance(settlement, dict):
        return 0.0
    cited = list(settlement.get('credited_facts') or []) + \
        list(settlement.get('justification_fact_ids') or [])
    cited = [c for c in cited if isinstance(c, str)]
    if not cited:
        return 0.0
    known = set(case['private_facts']) | set(case['shared_context'])
    bad = [c for c in cited
           if c not in known or (c in case['private_facts'] and c not in revealed)]
    return len(bad) / len(cited)


def score(case, settlement, revealed, norm=False, resolve=True):
    """Score one episode. `revealed`: fact ids observed to have been disclosed."""
    revealed = set(revealed or ())
    if resolve:
        settlement = resolve_provenance(case, settlement, revealed)
    ctx = build_context(settlement, revealed, norm=norm)
    C = eval_checks(case.get('content_checks'), ctx)
    P = eval_checks(case.get('provenance_checks'), ctx)

    flipped = set(flipped_checks(case))
    decisive = [d['fact_id'] for d in (case.get('decisive_facts') or [])]
    both = dict(C)
    both.update(P)

    all_c = all(C.values()) if C else False
    all_p = all(P.values()) if P else False
    # Content checks no decisive fact controls: ordinary settlement competence, kept
    # separable from the pooling signal.
    nonflip = [c for c in C if c not in flipped]
    close = (sum(C[c] for c in nonflip) / len(nonflip)) if nonflip \
        else (sum(C.values()) / max(1, len(C)))

    return {
        'close': close,
        'content': C,
        'provenance': P,
        'cbar': sum(C.values()) / max(1, len(C)),
        'pbar': sum(P.values()) / max(1, len(P)),
        'dca': sum(bool(both.get(c, False)) for c in flipped) / max(1, len(flipped)),
        'all_content': all_c,
        'all_prov': all_p,
        'joint': all_c and all_p,
        'disclosure_rate': sum(f in revealed for f in decisive) / max(1, len(decisive)),
        'schema_valid': schema_valid(case, settlement),
        'hallucinated_credit': hallucinated_credit(case, settlement, revealed),
        'settlement_resolved': settlement,
    }


def floor_score(case):
    """No dialogue, empty settlement. The denominator for normalised gain, and the test
    of whether a scenario is really a hidden-profile task at all."""
    return score(case, {}, set(), resolve=False)
