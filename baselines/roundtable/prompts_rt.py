"""Prompts for the round-table baseline. No acts, no scaffold, no planner.

Every agent gets the same instruction. The only thing that differs between them is what
they can SEE, which is the benchmark's whole point: `visible()` filters the scenario down
to each agent's own view, and the assertions below fail loudly if a private fact ever
reaches someone who was not given it.

That filtering is the one piece of machinery this arm has. Everything else is deliberately
absent, because the arm exists to answer "what happens if nobody steers?"
"""
import json

ORACLE_ONLY = ('content_checks', 'provenance_checks', 'acceptance_conditions',
               'decisive_facts')

# How the table converges on one answer.
DECIDERS = ('chair', 'vote', 'converge')


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


# ------------------------------------------------------------------ instructions
# One instruction for everyone. No role-specific coaching, no act menu, no reminder to
# elicit -- adding any of those would make this a method rather than a floor.
_TURN = ('You are in a meeting that must reach one shared decision. Say your piece in one '
         'short turn: contribute what you know that bears on the decision, respond to what '
         'others have said, and work towards agreement. Do not invent facts.')

_PROPOSE = ('\n\nThe meeting is ending. State your own view of the final decision now, as '
            'a JSON object matching this schema and nothing else: %s\n'
            'Use an empty string for anything the meeting did not settle. '
            "'credited_facts' and 'justification_fact_ids' list fact identifiers such as "
            'PF1 that you actually relied on.')

_CONVERGE = ('\n\nHere is the proposal currently on the table:\n%s\n\n'
             'If you agree with it, reply with exactly AGREE and nothing else. If you do '
             'not, reply with a corrected JSON object in the same schema and nothing '
             'else. Only object if something is wrong or missing.')


def agent_messages(case, conversation, agent_id, settling=False, proposal=None):
    """One turn for one agent. `settling` asks for a settlement instead of a remark."""
    head = ('Now enter the role-playing mode. You are taking part in a meeting chaired by '
            '%s. The participants are %s.\n\n%s'
            % (agent_of(case, case['decision_maker'])['name'], roster(case),
               render_persona(case, agent_id)))
    instr = _TURN
    if settling:
        instr = instr + _PROPOSE % json.dumps(case['settlement_schema'])
    if proposal is not None:
        instr = instr + _CONVERGE % json.dumps(proposal)

    for f in ORACLE_ONLY:
        assert f not in head and f not in instr, 'oracle field %r leaked to %s' % (f,
                                                                                   agent_id)
    mine = set(case['views'][agent_id])
    for fid, fact in case['private_facts'].items():
        if fid not in mine:
            assert fact['text'] not in head, '%s leaked to %s' % (fid, agent_id)

    return [{'role': 'system', 'content': head},
            {'role': 'USER', 'content': instr}] + list(conversation)


def settlement_messages(case, conversation):
    """Fallback extractor, used only when nobody emitted parseable JSON."""
    return [
        {'role': 'system', 'content': 'Extract the final settlement from the transcript.'},
        {'role': 'USER', 'content':
            'Return only a JSON object matching this schema, with no commentary: %s\n\n'
            'Use exactly the wording the meeting used. If a field was never settled, use '
            "an empty string. 'credited_facts' and 'justification_fact_ids' must list "
            'fact identifiers such as PF1 that the meeting actually relied on.\n\n'
            'Meeting: %s\nJSON: '
            % (json.dumps(case['settlement_schema']), transcript(conversation))}]


def to_chat(messages, speaker):
    """Map the neutral message list onto a chat template: the speaker's own prior turns
    become `assistant`, everyone else's become `user`."""
    out = [{'role': 'system', 'content': messages[0]['content']}]
    for m in messages[1:]:
        out.append({'role': 'assistant' if m['role'] == speaker else 'user',
                    'content': m['content']})
    return out
