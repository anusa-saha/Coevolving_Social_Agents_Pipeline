"""Lexical detectors: disclosure, leaks, addressing, and eliciting.

All are word-overlap rules with a frozen threshold. They are the measurement instrument
for every disclosure figure in this baseline, so the threshold (0.35) must not be tuned
after the fact -- it is calibrated once and reported.

`is_eliciting` is new here and has no analogue in the other two baselines. PPDPP and EPO
know whether a chair turn was an eliciting act because a planner chose the act. In
Sotopia-RL there is no planner -- the chair simply speaks -- so eliciting-ness has to be
read off the utterance itself. selftest.py measures this detector's agreement with the
human-facing act annotations, and that agreement is reported rather than assumed.
"""
import re

REVEAL_THRESHOLD = 0.35

_STOP = set('a an the is are was were be been being of to in on at for with and or but '
            'that this these those it its as by from we you i they he she them us our your '
            'has have had do does did not no yes will would can could should may might'.split())


def content_tokens(text):
    return [t for t in re.findall(r"[a-z0-9%$./-]+", (text or '').lower())
            if t not in _STOP and len(t) > 1]


def overlap(fact_text, utterance):
    """Share of the fact's content words that appear in the utterance."""
    ftok = set(content_tokens(fact_text))
    if not ftok:
        return 0.0
    return len(ftok & set(content_tokens(utterance))) / len(ftok)


def surname(name):
    return (name or '').split()[-1].lower() if name else ''


def addressed_in(utterance, agents, exclude=()):
    """Agent ids whose surname appears in the utterance."""
    low = (utterance or '').lower()
    out = set()
    for a in agents:
        if a['agent_id'] in exclude:
            continue
        s = surname(a['name'])
        if s and s in low:
            out.add(a['agent_id'])
    return out


_Q = re.compile(r'\?')
_ASKY = re.compile(r"\b(could you|can you|what|which|how|when|where|why|do you|are there|"
                   r"is there|please (?:confirm|clarify|specify|provide|share|walk)|"
                   r"tell me|any (?:constraints|issues|concerns|limits))\b", re.I)


def is_eliciting(utterance, agents, chair_id):
    """Did this chair turn ask a named participant for something?

    Two conditions, both required: it addresses at least one advisor by name, and it
    carries an interrogative. Requiring the name is what separates 'drawing information
    out of someone' from thinking aloud, and it matches how `addressed` is tracked.
    """
    if not utterance:
        return False
    named = addressed_in(utterance, agents, exclude={chair_id})
    if not named:
        return False
    return bool(_Q.search(utterance) or _ASKY.search(utterance))


def disclosures(case, speaker_id, utterance, already, threshold=REVEAL_THRESHOLD):
    """Private facts OWNED by this speaker that the utterance reveals."""
    out = []
    for fid, fact in case['private_facts'].items():
        if fid in already or fact['owner'] != speaker_id:
            continue
        if overlap(fact['text'], utterance) >= threshold:
            out.append(fid)
    return out


def leaks(case, speaker_id, utterance, already, threshold=REVEAL_THRESHOLD):
    """A speaker stating a fact it was never shown and nobody had yet disclosed.

    Runs on the CHAIR as well as advisors. In this baseline the chair's own weights are
    being trained against a reward that rises with disclosure, which makes fabricating a
    private fact a one-step reward hack; the chair's view provably contains no private
    fact, so the same rule applies to it unmodified.
    """
    view = case['views'].get(speaker_id, [])
    out = []
    for fid, fact in case['private_facts'].items():
        if fid in view or fid in already:
            continue
        if overlap(fact['text'], utterance) >= threshold:
            out.append(fid)
    return out


def assert_matches_ppdpp(verbose=True):
    """Fail loudly if ppdpp_csa/env.py's detector has drifted from this one.

    ppdpp_csa predates csa_core and still carries its own inline copy of the overlap
    rule, because it is vendored upstream code that the CSA port modified in place rather
    than a package written against this contract. That makes it the one duplicate left in
    the repo, so it is the one that needs watching: every disclosure number PPDPP reports
    is only comparable to the other arms while these two agree.

    Returns True when they agree, False when ppdpp_csa cannot be located (nothing to
    check), and raises AssertionError on genuine drift.
    """
    import os
    import sys

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cand = [os.path.join(here, 'ppdpp')]
    env_dir = next((d for d in cand if os.path.isfile(os.path.join(d, 'env.py'))), None)
    if env_dir is None:
        if verbose:
            print('[detectors] ppdpp_csa not present; drift check skipped')
        return False

    if env_dir not in sys.path:
        sys.path.insert(0, env_dir)
    try:
        import env as ppdpp_env
    except Exception as e:                           # noqa: BLE001
        if verbose:
            print('[detectors] could not import ppdpp_csa/env.py (%s); check skipped' % e)
        return False

    probes = [
        ('The insurer requires prior authorization by 3:00 PM today.',
         'Prior authorization has to be submitted by three this afternoon.'),
        ('Dalbavancin 1500 mg IV today and 1500 mg on day 8.',
         'We should approve dalbavancin at 1500 mg with a day-8 second dose.'),
        ('NorthStar Specialty Pharmacy is the contracted dispenser.',
         'Let us move on to the next agenda item.'),
        ('', 'anything at all'),
    ]
    theirs = getattr(ppdpp_env, '_overlap', None) or getattr(ppdpp_env, 'overlap', None)
    assert theirs is not None, 'ppdpp_csa/env.py exposes no overlap function'

    for fact, utt in probes:
        a, b = overlap(fact, utt), theirs(fact, utt)
        assert abs(a - b) < 1e-9, (
            'disclosure detector DRIFT between csa_core and ppdpp_csa/env.py\n'
            '  fact:      %r\n  utterance: %r\n  csa_core=%.6f  ppdpp_csa=%.6f\n'
            'Every disclosure figure in the comparison depends on these agreeing.'
            % (fact, utt, a, b))

    thr = getattr(ppdpp_env, 'REVEAL_THRESHOLD', REVEAL_THRESHOLD)
    assert abs(thr - REVEAL_THRESHOLD) < 1e-9, (
        'REVEAL_THRESHOLD differs: csa_core=%s ppdpp_csa=%s' % (REVEAL_THRESHOLD, thr))

    if verbose:
        print('[detectors] identical to ppdpp_csa/env.py (threshold %.2f)'
              % REVEAL_THRESHOLD)
    return True
