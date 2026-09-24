"""Prompts for SOTOPIA-Omega on CSA. No torch.

Omega's mechanism is: let the expert talk normally; when the dialogue STALLS, switch it
into a slow, multi-stage mode; keep the resulting corpus and fine-tune on it. The paper's
slow mode is a four-stage NEGOTIATION protocol (own utility -> opponent's utility ->
draft proposal -> confirm) because SOTOPIA is a negotiation benchmark.

CSA is not a negotiation. The chair and the advisors want the same outcome; the advisors
are not opponents holding leverage, they simply have not been asked. Estimating an
opponent's utility has no meaning when utilities are aligned. So the protocol is replaced
by a four-stage ELICITATION protocol with the same shape:

    Omega (negotiation, Thompson 2015 via the paper's C.1.3)
                                     Here (elicitation)
    1 Resource Assessment        ->  which decision fields are still unsupported
    2 Assessment of Difference   ->  whose ROLE makes them likely to hold the missing bit
    3 Initial Proposal           ->  draft the targeted question
    4 Update Proposal            ->  put the settlement on the record

Steps 1-3 are scaffolding calls here (stage_messages); step 4 is the chair's actual
utterance, not a fourth call. Note the asymmetry: Omega's step 4 REVISES the agent's own
position in response to the counterpart, which has no meaning when the parties' goals are
aligned -- so CSA's step 4 records a settlement instead. That is a real difference in the
protocol, not just a rename.

Keeping the mechanism and swapping the protocol is the point. Forcing CSA into a
negotiation frame would be adapting the dataset to the method.

The stage outputs are SCAFFOLDING, not corpus. Only the final utterance enters the
dialogue and the SFT labels, so the student learns to produce it directly. That is where
the distillation actually happens.
"""
import json

ORACLE_ONLY = ('content_checks', 'provenance_checks', 'acceptance_conditions',
               'decisive_facts')

MODES = ('fast', 'slow')          # Omega's easy / hard
OPPONENTS = ('none', 'withhold')


def agent_of(case, agent_id):
    return next(a for a in case['agents'] if a['agent_id'] == agent_id)


def roster(case):
    return ', '.join('%s (%s)' % (a['name'], a['role']) for a in case['agents'])


def advisors_of(case):
    return [a for a in case['agents'] if a['agent_id'] != case['decision_maker']]


def visible(case, agent_id):
    keys = case['views'][agent_id]
    shared = {k: v for k, v in case['shared_context'].items() if k in keys}
    private = {k: v['text'] for k, v in case['private_facts'].items() if k in keys}
    return shared, private


def render_persona(case, agent_id):
    shared, private = visible(case, agent_id)
    a = agent_of(case, agent_id)
    lines = ['You are %s, %s.' % (a['name'], a['role']),
             'Scenario: %s' % case['description'],
             'What everyone in the meeting knows:']
    lines += ['  %s: %s' % (k, v) for k, v in sorted(shared.items())]
    if private:
        lines.append('What only you know (the others cannot see this; share it when it '
                     'bears on the decision):')
        lines += ['  %s: %s' % (k, v) for k, v in sorted(private.items())]
    out = '\n'.join(lines)
    if agent_id == case['decision_maker']:
        for fid, fact in case['private_facts'].items():
            assert fact['text'] not in out, '%s leaked into the chair persona' % fid
    for f in ORACLE_ONLY:
        assert f not in out, 'oracle field %r leaked into a persona' % f
    return out


def transcript(conversation, limit=None):
    turns = conversation[-limit:] if limit else conversation
    return '\n'.join('%s: %s' % (t['role'], t['content']) for t in turns)


_SETTLE = ('\nRecord the settlement now. The settlement schema is: %s\n'
           'Reply with the JSON object only.')

_FAST = ('You lead this discussion and you alone will record the final settlement. The '
         'others hold information you do not have; it is your job to draw it out. Please '
         'reply with only one short and succinct turn.')


def chair_messages(case, conversation, settling=False, plan=''):
    """The chair's prompt. `plan` is the slow-mode output; empty means fast mode."""
    dm = case['decision_maker']
    head = ('Now enter the role-playing mode. You are chairing a meeting that must reach '
            'one decision. The other participants are %s.\n\n%s'
            % (roster(case), render_persona(case, dm)))
    instr = _FAST
    if plan:
        instr += ('\n\nYou have just worked out where this meeting is stuck:\n%s\n\n'
                  'Act on that now. Address one named participant and ask for the '
                  'specific thing that is missing.' % plan.strip())
    if settling:
        instr += _SETTLE % json.dumps(case['settlement_schema'])
    return [{'role': 'system', 'content': head},
            {'role': 'USER', 'content': instr}] + list(conversation)


# ------------------------------------------------------------------ slow mode
_STAGE_HEAD = """You are %s, chairing a meeting that must reach one decision.

The meeting has stalled: several turns have passed and nobody has told you anything you
did not already know. Work out why, one step at a time.

The decision must fill these fields:
%s

Participants other than you:
%s

Meeting so far:
%s
"""

_S1 = """%s
STEP 1. List the decision fields for which NOTHING said so far gives you a basis to
choose. One per line, field name then why it is unsupported. At most four lines."""

_S2 = """%s
Unsupported fields:
%s

STEP 2. For each unsupported field, name the ONE participant whose role makes them most
likely to hold the missing detail, and say what that detail probably is. Reason from
their job title and from what they have not said. Do not invent facts. One line each."""

_S3 = """%s
Unsupported fields:
%s

Who likely holds what:
%s

STEP 3. Write the single most valuable question to ask next. Address one participant by
name and ask for a specific value, limit, date or constraint -- not a general update.
One sentence. Nothing else."""


def stage_messages(case, conversation, stage, unsupported='', holders='', window=10):
    """Slow-mode scaffolding. These outputs never enter the transcript.

    The leak check below guards against a fact's text getting baked into the STATIC
    scaffolding (the fields list, the roster, the step instructions) -- e.g. an oracle
    field or a private fact substituted into the template by mistake. It is not a check
    against the fact having been legitimately disclosed by an advisor and sitting in the
    conversation transcript: that's expected, since the transcript is quoted verbatim
    into `body` (via _STAGE_HEAD) and eliciting private facts into the dialogue is the
    whole point of CSA. So a fact is only flagged as leaked if it appears in `body`
    through something OTHER than the transcript.
    """
    dm = case['decision_maker']
    fields = '\n'.join('  - %s' % k
                       for k in (case['settlement_schema'].get('decisions') or {}))
    others = '\n'.join('  %s (%s)' % (a['name'], a['role']) for a in advisors_of(case))
    convo_text = transcript(conversation, window)
    head = _STAGE_HEAD % (agent_of(case, dm)['name'], fields, others, convo_text)
    table = {1: _S1 % head,
             2: _S2 % (head, unsupported.strip() or '(none identified)'),
             3: _S3 % (head, unsupported.strip() or '(none identified)',
                       holders.strip() or '(none identified)')}
    if stage not in table:
        raise ValueError('slow mode has three scaffolding stages (1-3); step 4 of the '
                         'protocol is the spoken chair turn, not a stage call. '
                         'Got stage=%r.' % (stage,))
    body = table[stage]
    out = [{'role': 'system', 'content': 'You reason carefully and answer briefly.'},
           {'role': 'USER', 'content': body}]
    for fid, fact in case['private_facts'].items():
        text = fact['text']
        if text and text in body and text not in convo_text:
            assert False, \
                '%s leaked into slow-mode stage %d' % (fid, stage)
    return out


def format_plan(unsupported, holders, question):
    parts = []
    if unsupported.strip():
        parts.append('Still unsupported:\n%s' % unsupported.strip())
    if holders.strip():
        parts.append('Likely holders:\n%s' % holders.strip())
    if question.strip():
        parts.append('Ask next: %s' % question.strip())
    return '\n'.join(parts)


# ------------------------------------------------------------------ advisors
_ADVISOR = ('You are advising the chair, who cannot see what only you know. Please reply '
            'with only one short and succinct turn. Answer direct questions truthfully. '
            'Are you ready to play the game?')

# The adversarial variant. Deliberately vague rather than a flat refusal: a refusal is
# trivially detectable and would make the arm a test of nothing. This withholds substance
# while staying in role, which is what makes the disclosure detector's margin the thing
# under test.
_WITHHOLD = ('You are advising the chair. You are cautious about volunteering specifics: '
             'answer in general terms, defer where you can, and do not offer exact '
             'figures, dates or limits unless the chair asks you for that exact detail '
             'directly and by name. Never state something untrue. Please reply with only '
             'one short and succinct turn. Are you ready to play the game?')


def advisor_messages(case, conversation, agent_id, opponent=None):
    dm = case['decision_maker']
    head = ('Now enter the role-playing mode. You are taking part in a meeting chaired by '
            '%s. The participants are %s.\n\n%s'
            % (agent_of(case, dm)['name'], roster(case), render_persona(case, agent_id)))
    instr = _WITHHOLD if (opponent and agent_id == opponent) else _ADVISOR
    return [{'role': 'system', 'content': head},
            {'role': 'USER', 'content': instr},
            {'role': agent_of(case, agent_id)['name'],
             'content': "Yes, I'm ready to play the game!"}] + list(conversation)


def settlement_messages(case, conversation):
    return [
        {'role': 'system', 'content': 'Extract the final settlement from the transcript.'},
        {'role': 'USER', 'content':
            'Return only a JSON object matching this schema, with no commentary: %s\n\n'
            'Use exactly the wording the chair used. If the chair never settled a field, '
            "use an empty string. 'credited_facts' and 'justification_fact_ids' must list "
            'fact identifiers such as PF1 that the chair actually relied on.\n\n'
            'Meeting: %s\nJSON: '
            % (json.dumps(case['settlement_schema']), transcript(conversation))}]


def to_chat(messages, speaker):
    out = [{'role': 'system', 'content': messages[0]['content']}]
    for m in messages[1:]:
        out.append({'role': 'assistant' if m['role'] == speaker else 'user',
                    'content': m['content']})
    return out