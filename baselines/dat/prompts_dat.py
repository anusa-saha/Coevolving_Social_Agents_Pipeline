"""Prompt construction for the DAT port. No torch: importable on a CPU-only box.

There is deliberately almost nothing here.

DAT's whole claim is that the language model is frozen and the prompt is untouched: the
only thing that changes between the unsteered and the steered agent is L continuous
prefix embeddings prepended to the *same* token embeddings. So this module borrows
ppdpp/prompt.py's builder and adds no instruction of its own -- if the DAT chair prompt
differed from the control's by even a sentence, a gain could be attributed to the wording
rather than to the steering, and the arm would measure nothing.

Two calls into the shared builder, both already used by other arms:

    non-settling turn   CSAMessages(case, 'system', conv, action=None)
                        -- the no-planner control, exactly as PPDPP's own docstring
                           describes it
    settling turn       CSAMessages(case, 'system', conv, action='decide')
                        -- byte-identical to PPDPP's settling prompt, so the settlement
                           JSON is requested the same way in every arm

The settling turn is chosen by the ENVIRONMENT (the chair's last slot in the turn order),
not by a planner, which is the same rule Sotopia-ToM uses. DAT has no discrete acts to
choose with: its action is a vector in R^d', so there is nothing that could select
'decide'. Holding that rule fixed across the arm's three conditions keeps the comparison
internal to the steering.
"""
import paths  # noqa: F401  -- puts the repo root and ppdpp/ on sys.path

from prompt import (CSAAct, CSA_ELICITING_ACTS, CSA_ORACLE_ONLY,   # noqa: F401
                    CSA_SETTLING_ACT, CSAMessages, _csa_render, _csa_roster,
                    qwen_prompt)

ORACLE_ONLY = CSA_ORACLE_ONLY


def chair_messages(case, conversation, settling=False):
    """The chair prompt. No DAT-specific text anywhere in it -- that is the point."""
    return CSAMessages(case, 'system', conversation,
                       action=(CSA_SETTLING_ACT if settling else None))


def advisor_messages(case, conversation, agent_id):
    return CSAMessages(case, 'user', conversation, agent_id=agent_id)


def settlement_messages(case, conversation):
    """Fallback extractor, used only when the chair never emitted parseable JSON."""
    return CSAMessages(case, 'settlement', conversation)


def to_chat(messages, role):
    """PPDPP's message list -> ChatML roles for the tokenizer's own chat template."""
    return qwen_prompt(messages, role)


def assert_filtered(case, messages, who='chair'):
    """No private fact and no oracle field may reach this prompt.

    Called by selftest.py over every scenario rather than on the hot path. The chair is
    the surface that matters: DAT's reward rises with disclosure, so a private fact
    visible in the chair's own prompt would let the planner score an elicitation it never
    performed.
    """
    blob = ' '.join(m.get('content') or '' for m in messages)
    for field in ORACLE_ONLY:
        assert field not in blob, 'oracle field %r leaked into the %s prompt' % (field, who)
    if who == 'chair':
        for fid, fact in case['private_facts'].items():
            assert fact['text'] not in blob, '%s leaked into the chair prompt' % fid
    return True
