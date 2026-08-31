"""Chair prompting strategies from Sotopia-ToM, plus the frozen advisor prompt.

Reimplemented FROM THE PAPER (arXiv:2605.02307), not from the authors' code: the paper's
"Code" link points at the generic sotopia framework, and the reproducibility statement
says the Sotopia-ToM codebase "will be made publicly available". It was not, at the time
of writing. Treat every template here as our reading of the paper's description.

Five arms. The first is not in the paper -- it exists because CSA's own chair prompt
already contains an explicit elicitation instruction, which is most of what CoT and ToM
scaffolding are meant to induce. Without a stripped control the four real arms would very
likely land on top of each other and the study would measure nothing.

    stripped   no elicitation instruction at all          <- control, ours
    basic      Sotopia-ToM vanilla                        <- their baseline
    cot        CoT-Elicitation (their CoT-Privacy, repointed)
    tom_coach  stateless mental-state table, injected
    tom_belief stateful belief tracker, updated per turn

CoT-Privacy is renamed CoT-Elicitation and its leakage-check step is repointed at
"which decisive fact is still missing". On CSA there is nothing to withhold -- every
private fact should be pooled -- so a leakage check would be a no-op that quietly wastes
the strategy's reasoning budget. This is a deliberate, documented departure.
"""
import json

ORACLE_ONLY = ('content_checks', 'provenance_checks', 'acceptance_conditions',
               'decisive_facts')

STRATEGIES = ('stripped', 'basic', 'cot', 'tom_coach', 'tom_belief')


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
_SETTLE = ('\nRecord the settlement now. The settlement schema is: %s\n'
           'Reply with the JSON object only.')

# The control. Deliberately says nothing about drawing information out -- this is what
# Sotopia-ToM's vanilla agent is told, and it is the arm the interventions must beat.
_STRIPPED = ('You are chairing this meeting and you alone will record the final '
             'settlement. Please reply with only one short and succinct turn.')

# Their vanilla: pursue the objective, engage naturally, exit when done. The channel
# clause is dropped -- CSA has no private channel.
_BASIC = ('You are chairing this meeting and you alone will record the final settlement. '
          'Pursue the decision, engage naturally with the other participants, and close '
          'once you have what you need. Please reply with only one short and succinct '
          'turn.')

_COT = """You are chairing this meeting and you alone will record the final settlement.

Before you speak, think through these steps in order, then give your turn.

1. Goal progress: what has the meeting settled so far, and what remains open?
2. Information gaps: which decision fields are still unsupported by anything said?
3. Who holds it: which named participant is most likely to hold the missing detail?
4. Missing-fact check: is there a constraint nobody has stated that would change the
   decision if it existed?
5. Act: choose the single most useful thing to say now.

Reply in exactly this form:
THINKING: <your reasoning, at most four short lines>
TURN: <one short and succinct turn, spoken to the meeting>"""

# The two ToM arms differ in what the note IS, so they differ in how it is framed. Coach
# hands the chair a fresh read of the room; Belief hands it something the chair has been
# maintaining all meeting. Wrapping both identically would make the arms differ only in a
# hidden preprocessing step, which is not what the paper describes.
_TOM_COACH_HEAD = """You are chairing this meeting and you alone will record the final settlement.

An analyst has just read the meeting and assessed what each participant is likely
thinking and what they may know that you have not yet been told. This is a fresh read of
the current moment.

%s

Use it to decide what to say. Please reply with only one short and succinct turn."""

_TOM_BELIEF_HEAD = """You are chairing this meeting and you alone will record the final settlement.

You have been keeping track of this meeting since it began. This is your running belief
state -- what you think each participant holds, what is settled, what is still open, and
the question you judged most valuable to ask next. It carries forward everything you have
inferred so far, not just the last exchange.

%s

Act on it. Please reply with only one short and succinct turn."""

# The separate call that builds the mental-state table for ToM-Coach and ToM-Belief.
_TOM_ANALYST = """You analyse what participants in a meeting are thinking.

The chair is %s. The other participants are:
%s

Meeting so far:
%s

For each participant other than the chair, give one line in exactly this form:
  <name> | likely holds: <what job-specific information they plausibly have that has not
  been said> | signals: <anything in their turns suggesting they are holding back or
  waiting to be asked>

Be concrete and short. Do not invent facts; reason from their role and from what they
have and have not said. At most one line per participant, no preamble."""

_BELIEF_UPDATE = """You maintain a running belief state about a meeting.

Previous belief state:
%s

New turns since the last update:
%s

Return an updated belief state as JSON with exactly these keys:
  "agents":   object mapping each non-chair participant name to a short string describing
              what they are believed to hold and whether they have disclosed it
  "settled":  array of short strings, decision points that now appear settled
  "open":     array of short strings, what still needs to be established
  "ask_next": string, the single most valuable question to ask next and of whom

Update incrementally: carry forward what has not changed. Return the JSON object only."""


# ------------------------------------------------------------------ chair prompts
def chair_messages(case, conversation, strategy='basic', settling=False, tom_note=''):
    """Messages for the chair under one strategy.

    `tom_note` carries the analyst output (tom_coach) or the belief state (tom_belief);
    it is ignored by the other arms.
    """
    assert strategy in STRATEGIES, strategy
    dm = case['decision_maker']
    head = ('Now enter the role-playing mode. You are chairing a meeting that must reach '
            'one decision. The other participants are %s.\n\n%s'
            % (roster(case), render_persona(case, dm)))

    if strategy == 'stripped':
        instr = _STRIPPED
    elif strategy == 'basic':
        instr = _BASIC
    elif strategy == 'cot':
        instr = _COT
    elif strategy == 'tom_coach':
        instr = _TOM_COACH_HEAD % (tom_note.strip() or '(no analysis available this turn)')
    else:
        instr = _TOM_BELIEF_HEAD % (tom_note.strip() or '(belief state still empty)')

    if settling:
        instr = instr + _SETTLE % json.dumps(case['settlement_schema'])
        if strategy == 'cot':
            # a THINKING block would corrupt the JSON object the verifier parses
            instr += '\nDo not include a THINKING line on this turn.'

    return [{'role': 'system', 'content': head},
            {'role': 'USER', 'content': instr}] + list(conversation)


def analyst_messages(case, conversation, window=8):
    dm = case['decision_maker']
    others = '\n'.join('  %s (%s)' % (a['name'], a['role']) for a in case['agents']
                       if a['agent_id'] != dm)
    return [{'role': 'system', 'content': 'You analyse participants in a meeting.'},
            {'role': 'USER', 'content': _TOM_ANALYST
             % (agent_of(case, dm)['name'], others, transcript(conversation, window))}]


def belief_messages(case, prev_state, new_turns):
    prev = json.dumps(prev_state, indent=1) if prev_state else '(none yet)'
    return [{'role': 'system', 'content': 'You maintain a belief state as JSON.'},
            {'role': 'USER', 'content': _BELIEF_UPDATE
             % (prev, transcript(new_turns) or '(no new turns)')}]


def render_belief(state):
    """Belief state -> the note injected into the chair prompt."""
    if not isinstance(state, dict) or not state:
        return ''
    lines = []
    for name, desc in (state.get('agents') or {}).items():
        lines.append('  %s: %s' % (name, desc))
    out = []
    if lines:
        out.append('What each participant is believed to hold:\n' + '\n'.join(lines))
    if state.get('open'):
        out.append('Still open: ' + '; '.join(str(x) for x in state['open'][:5]))
    if state.get('ask_next'):
        out.append('Most valuable next question: %s' % state['ask_next'])
    return '\n'.join(out)


# ------------------------------------------------------------------ advisors
def advisor_messages(case, conversation, agent_id):
    """Frozen throughout. Identical to the other baselines so the environment half of
    the comparison is unchanged."""
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


def split_thinking(text):
    """CoT arm emits THINKING/TURN. Return the spoken turn only -- the reasoning must not
    enter the transcript, or advisors would read the chair's private deliberation."""
    if 'TURN:' in text:
        return text.split('TURN:', 1)[1].strip()
    # model ignored the format; strip a leading THINKING block if there is one
    if text.strip().upper().startswith('THINKING:'):
        parts = text.split('\n')
        keep = [p for p in parts[1:] if p.strip()]
        return '\n'.join(keep).strip() or text.strip()
    return text.strip()
