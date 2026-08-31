"""Prompt construction. No torch, importable anywhere.

Unlike the planner baselines there is no act instruction anywhere in this file. The chair
is the policy being trained, so its prompt carries persona, view and transcript and
nothing else -- what it decides to do is what the weights encode.

Every participant prompt is filtered through case['views']. ORACLE_ONLY names the four
fields that must never reach any prompt; render_persona asserts it.
"""
import json

# Answer key. None of these may appear in any prompt shown to a participant.
ORACLE_ONLY = ('content_checks', 'provenance_checks', 'acceptance_conditions',
               'decisive_facts')


def agent_of(case, agent_id):
    return next(a for a in case['agents'] if a['agent_id'] == agent_id)


def roster(case):
    return ', '.join('%s (%s)' % (a['name'], a['role']) for a in case['agents'])


def visible(case, agent_id):
    keys = case['views'][agent_id]
    shared = {k: v for k, v in case['shared_context'].items() if k in keys}
    private = {k: v['text'] for k, v in case['private_facts'].items() if k in keys}
    return shared, private


def render_persona(case, agent_id):
    """The persona block for one agent, filtered to that agent's view."""
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


# ------------------------------------------------------------------ chair (the policy)
def chair_messages(case, conversation, settling=False):
    """The prompt the trained policy sees. No act, no strategy, no planner."""
    dm = case['decision_maker']
    head = ('Now enter the role-playing mode. You are chairing a meeting that must reach '
            'one decision. The other participants are %s.\n\n%s'
            % (roster(case), render_persona(case, dm)))
    instr = ('You lead this discussion and you alone will record the final settlement. '
             'The others hold information you do not have; it is your job to draw it out. '
             'Please reply with only one short and succinct turn.')
    if settling:
        instr += ('\nRecord the settlement now. The settlement schema is: %s\n'
                  'Reply with the JSON object only.'
                  % json.dumps(case['settlement_schema']))
    msgs = [{'role': 'system', 'content': head},
            {'role': 'USER', 'content': instr}]
    msgs.extend(conversation)
    return msgs


# ------------------------------------------------------------------ advisors (frozen)
def advisor_messages(case, conversation, agent_id):
    dm = case['decision_maker']
    head = ('Now enter the role-playing mode. You are taking part in a meeting chaired by '
            '%s. The participants are %s.\n\n%s'
            % (agent_of(case, dm)['name'], roster(case), render_persona(case, agent_id)))
    instr = ('You are advising the chair, who cannot see what only you know. Please reply '
             'with only one short and succinct turn. Answer direct questions truthfully. '
             'Are you ready to play the game?')
    return [{'role': 'system', 'content': head},
            {'role': 'USER', 'content': instr},
            {'role': agent_of(case, agent_id)['name'],
             'content': "Yes, I'm ready to play the game!"}] + list(conversation)


# ------------------------------------------------------------------ fallback extractor
def settlement_messages(case, conversation):
    """Used only when the chair never emitted parseable JSON. Structured extraction, not
    judgement: it scores nothing."""
    return [
        {'role': 'system', 'content': 'Extract the final settlement from the transcript.'},
        {'role': 'USER', 'content':
            'Return only a JSON object matching this schema, with no commentary: %s\n\n'
            'Use exactly the wording the chair used. If the chair never settled a field, '
            "use an empty string. 'credited_facts' and 'justification_fact_ids' must list "
            'fact identifiers such as PF1 that the chair actually relied on.\n\n'
            'Meeting: %s\nJSON: '
            % (json.dumps(case['settlement_schema']), transcript(conversation))}]


# ------------------------------------------------------------------ chat rendering
def to_chat(messages, speaker):
    """PPDPP-style message list -> ChatML roles: the named speaker becomes the assistant,
    everyone else the user."""
    out = [{'role': 'system', 'content': messages[0]['content']}]
    for m in messages[1:]:
        out.append({'role': 'assistant' if m['role'] == speaker else 'user',
                    'content': m['content']})
    return out
