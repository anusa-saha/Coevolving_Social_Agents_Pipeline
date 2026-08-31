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
