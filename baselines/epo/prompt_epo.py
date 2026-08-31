"""Prompt construction for the EPO port. No torch: importable on a CPU-only box.

Three surfaces:

  strategist_messages  -- NEW. The prompt LLM_s is trained on. View-filtered to the
                          chair, so the strategist can never see a private fact. This
                          has no analogue in vanilla EPO, whose environments have no
                          hidden profiles and therefore nothing to filter.
  chair_messages       -- the chair prompt, with `CSAAct[tau] + sigma` where PPDPP had
                          `CSAAct[action]` alone.
  advisor / settlement / acceptance -- reused unchanged from ppdpp_csa.prompt.
"""
import json
import re

import config                                        # noqa: F401  (sys.path shim)
from prompt import (CSAAct, CSA_ELICITING_ACTS, CSA_ORACLE_ONLY, CSA_SETTLING_ACT,
                    CSAMessages, _csa_agent, _csa_render, _csa_roster, chatgpt_prompt,
                    qwen_prompt)

ACTS = sorted(CSAAct)                                # ask, decide, followup, share

STRATEGIST_SYSTEM = """You plan the chair's next move in a meeting that must reach one decision.

Reply with exactly one line, in this form:
  <act>: <strategy>

<act> is one of: ask, followup, share, decide.
  ask       open a topic nobody has raised, with a named participant
  followup  push on an answer already given: get the exact detail it left vague
  share     relay what one participant disclosed, so the others can react
  decide    record the settlement now

<strategy> is a single concrete instruction, at most 20 words. Name people where it helps.
Write nothing else: no explanation, no quotes, no second line.

%s

The other participants hold information you cannot see. Your job is to draw it out
before the decision is recorded."""


# ------------------------------------------------------------------ parsing
# Leading and TRAILING decoration both have to be eaten: small models wrap the tag as
# "**share**:" often enough that leaving the trailing "**" in sigma is a real defect.
_TAG = re.compile(r'^\s*[\*\-\s"\']*(%s)\b[\*"\'\s]*[:\-—]?\s*' % '|'.join(ACTS), re.I)

# Stricter: the separator is REQUIRED. clean_sigma uses this so that a sigma which
# legitimately begins with an act word ("ask Morgan about the lane limits") keeps it,
# while a duplicated tag ("followup: press ...") is removed.
_TAG_SEP = re.compile(r'^\s*[\*\-\s"\']*(%s)\b[\*"\'\s]*[:\-—]\s*' % '|'.join(ACTS), re.I)


def parse_strategy(text, default='ask'):
    """'followup: press Patel for the exact clearance' -> ('followup', 'press ...').

    The tag is not decoration: three sites in the environment branch on it (settlement
    token budget, addressed-advisor tracking, elicited-vs-volunteered). A miss is
    recorded by the caller rather than silently absorbed, because a policy that stops
    emitting parseable tags is a training failure, not a formatting quirk.
    """
    raw = (text or '').strip()
    # Models sometimes emit a preamble line; the tag is what matters, so scan lines.
    for line in [raw] + raw.splitlines():
        m = _TAG.match(line)
        if m:
            tau = m.group(1).lower()
            sigma = line[m.end():].strip().strip('"\'').rstrip('.')
            return tau, sigma, True
    # Last resort: a bare act word anywhere.
    low = raw.lower()
    for a in ACTS:
        if re.search(r'\b%s\b' % a, low):
            return a, raw.strip(), False
    return default, raw.strip(), False


def format_strategy(tau, sigma):
    return '%s: %s' % (tau, sigma.strip()) if sigma else '%s:' % tau


# ------------------------------------------------------------------ strategist
def strategist_messages(case, conversation, prior_acts=(), max_turns=12):
    """The prompt LLM_s sees. Chair's view only.

    `prior_acts` is EPO's a_{1:t-1} term -- cheap to include and it stops the policy
    repeating the same act every turn.

    The assertion is not paranoia: the strategist is trained against a reward that rises
    with disclosure, so any leak of a private fact into THIS prompt is a one-step reward
    hack that would score as a successful elicitation.
    """
    dm = case['decision_maker']
    head = STRATEGIST_SYSTEM % _csa_render(case, dm)

    for field in CSA_ORACLE_ONLY:
        assert field not in head, 'oracle field %r leaked into the strategist prompt' % field
    for fid, fact in case['private_facts'].items():
        assert fact['text'] not in head, '%s leaked into the strategist prompt' % fid

    turns = conversation[-max_turns:]
    body = '\n'.join('%s: %s' % (t['role'], t['content']) for t in turns)
    if prior_acts:
        body += '\n\nStrategies so far: %s' % ', '.join(prior_acts)
    body += '\n\nNext strategy:'

    return [{'role': 'system', 'content': head},
            {'role': 'user', 'content': body}]


# ------------------------------------------------------------------ chair
def chair_messages(case, conversation, tau, sigma):
    """PPDPP appends CSAAct[action]; EPO appends CSAAct[tau] plus the open strategy.

    Built on ppdpp_csa's CSAMessages so the persona block, view filtering and settlement
    schema injection stay identical -- only the instruction tail differs.
    """
    msgs = CSAMessages(case, 'system', conversation, action=tau)
    if sigma:
        # msgs[1] is the instruction turn; CSAAct[tau] is already the last thing in it,
        # except on a settling turn where the schema follows and must stay last.
        instr = msgs[1]['content']
        if tau == CSA_SETTLING_ACT:
            head, _, schema = instr.partition('\nThe settlement schema is:')
            msgs[1]['content'] = '%s %s\nThe settlement schema is:%s' % (head, sigma, schema)
        else:
            msgs[1]['content'] = '%s %s' % (instr, sigma)
    return msgs


def advisor_messages(case, conversation, agent_id):
    return CSAMessages(case, 'user', conversation, agent_id=agent_id)


def settlement_messages(case, conversation):
    return CSAMessages(case, 'settlement', conversation)


def critic_messages(case, conversation):
    return CSAMessages(case, 'critic', conversation)


# ------------------------------------------------------------------ stage 1
VERBALISE_SYSTEM = (
    'You state the instruction a meeting chair was following. You are given the act it '
    'performed; you only have to write the instruction. Answer with one line and nothing '
    'else.')

VERBALISE_TEMPLATE = """Meeting so far:
{context}

The chair ({chair}) then said:
"{utterance}"

The act is already known to be: {tau}

Write, in at most 15 words, the instruction the chair was following. Write it as a
directive addressed TO the chair ("ask X about Y", "press X for the exact Z"), not as a
summary of what they said. Name people where the chair named them. No quotes, no
preamble, no trailing full stop.

Instruction:"""


def verbalise_prompt(context, chair, utterance, tau):
    return VERBALISE_TEMPLATE.format(context=context[-2500:], chair=chair,
                                     utterance=utterance[:800], tau=tau)


def clean_sigma(text, max_words=20):
    """Strip the ways a small model wraps a one-line answer."""
    s = (text or '').strip()
    s = re.sub(r'^(instruction|strategy|answer)\s*[:\-]\s*', '', s, flags=re.I)
    s = s.strip().strip('"\'').strip()
    s = s.split('\n')[0].strip()
    # A verbaliser that echoes the tag would double it once format_strategy runs. Only
    # strip a tag that carries its separator -- "ask Morgan about X" is a valid sigma.
    s = _TAG_SEP.sub('', s).strip()
    words = s.split()
    if len(words) > max_words:
        s = ' '.join(words[:max_words])
    return s.rstrip('.').strip()
