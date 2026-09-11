"""Label chair turns with one of four intents, using an annotator model via OpenRouter.

Only turns marked annotate=True are labelled -- the closing settlement turn was
instructed, so its intent is known and including it would inflate the DECIDE class.

Every call is logged to a JSONL file as it happens: prompt context, raw response,
parsed label, latency, and two mechanical cross-checks that need no human. Those checks
give partial ground truth for two of the four classes, so the annotator can be audited
before any hand-labelling:

  DECIDE  -- the turn should contain a settlement (parseable JSON or decision values)
  SHARE   -- the turn should repeat a fact disclosed earlier in the conversation

The key is read from OPENROUTER_API_KEY. Never put it in this file: it is git-tracked.

    python annotate_intents.py --in conversations/conversations-train.jsonl
"""
import argparse
import io
import json
import os
import sys
import re
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from csa_core import paths  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from prompt import CSAAct                                    # noqa: E402
from utils import load_dataset                               # noqa: E402

LABELS = sorted(CSAAct.keys())          # ask, decide, followup, share

SYSTEM = (
    "You label one turn of a meeting with the intent the chair was pursuing. "
    "Answer with exactly one word: ask, followup, share, or decide. Nothing else."
)

# Worked examples matter more than definitions here. The failure mode without them is
# that any turn formatted as a list of decisions gets called `decide`, even when the
# chair is only repeating back what an advisor just told it -- which is `share`.
FEWSHOT = """Here are four labelled examples.

--- Example 1 ---
Meeting so far:
Meeting: The group convenes to decide tomorrow's dock schedule.
Chair: Let us work out the receiving order.
The chair then said:
"CW2 Foster, what handling constraints apply to the clinic consumables?"
Intent: ask
Why: it opens a topic nobody has raised, addressed to a named person.

--- Example 2 ---
Meeting so far:
CW2 Foster: The clinic consumables need cold-chain handling on arrival.
The chair then said:
"You said cold-chain -- what exact temperature range, and for how long?"
Intent: followup
Why: it refers back to an answer already given and demands the missing detail.

--- Example 3 ---
Meeting so far:
CW2 Foster: The clinic consumables need cold-chain handling and must dock by 1000.
Lt. Morgan: The dock is clear all morning.
The chair then said:
"Thank you, CW2 Foster. To summarise what we have: the clinic consumables need
cold-chain handling and a slot by 1000. Lt. Morgan, does that work against your
dock availability?"
Intent: share
Why: it repeats what one participant disclosed so the others can react. It is NOT
decide -- no decision is being recorded, only relayed.

--- Example 4 ---
Meeting so far:
CW2 Foster: Cold-chain, dock by 1000.
Lt. Morgan: The dock is clear.
The chair then said:
"{\\"decisions\\": {\\"clinic_dock_time\\": \\"0900-1000\\", \\"handling\\": \\"cold-chain\\"}}"
Intent: decide
Why: it records the settlement itself.
"""

TEMPLATE = """{fewshot}
Intent definitions:
  ask       - asks a named participant about a topic NOT yet raised in the meeting
  followup  - pushes on an answer ALREADY given: demands the exact detail, number or
              date it left vague, or contests it
  share     - repeats or relays what a participant disclosed, so the others can react.
              A turn that lists back what people said, without recording a final
              settlement, is share -- not decide.
  decide    - records or proposes the settlement itself, as decision values

Priority if a turn fits more than one: decide > share > followup > ask.

--- Now label this one ---
Meeting so far:
{context}

The chair ({chair}) then said:
"{utterance}"

Intent:"""


def parse_label(text):
    low = str(text).strip().lower()
    for lab in LABELS:                       # exact or leading match first
        if low == lab or low.startswith(lab):
            return lab
    for lab in LABELS:                       # then anywhere
        if re.search(r'\b%s\b' % lab, low):
            return lab
    return None


def check_decide(turn_text):
    """A DECIDE turn should carry a settlement: JSON, or explicit decision language."""
    if '{' in turn_text and '}' in turn_text:
        return True
    return bool(re.search(r'\b(we will|let\'s go with|assign|approved|final|record)\b',
                          turn_text.lower()))


def _repeats(turn_text, prior_texts, thr=0.25):
    from env import _content_tokens
    ut = set(_content_tokens(turn_text))
    for txt in prior_texts:
        ft = set(_content_tokens(txt))
        if ft and len(ft & ut) / len(ft) >= thr:
            return True
    return False


def check_share(turn_text, prior_any, prior_decisive):
    """Two separate questions, deliberately not collapsed.

    `label_ok`  -- does the turn repeat ANY prior participant content? That is what
                   validates the SHARE label itself.
    `pools`     -- does it repeat a DECISIVE private fact? That is the behaviour the
                   benchmark is about, and a turn can be correctly labelled SHARE while
                   relaying only shared context everyone already had -- which is the
                   classic hidden-profile failure (Stasser and Titus, 1985).
    """
    return _repeats(turn_text, prior_any), _repeats(turn_text, prior_decisive)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--in', dest='infile', required=True)
    p.add_argument('--out', default='')
    p.add_argument('--log', default='')
    p.add_argument('--model', default=paths.ANNOTATOR_MODEL)
    p.add_argument('--base_url', default='https://openrouter.ai/api/v1')
    p.add_argument('--context_turns', type=int, default=6)
    p.add_argument('--max_retries', type=int, default=6)
    p.add_argument('--max_tokens', type=int, default=24,
                   help='8 was too tight: the model returned <pad> instead of a word.')
    p.add_argument('--sleep', type=float, default=0.0,
                   help='Fixed pause between calls, for tight free-tier rate limits.')
    p.add_argument('--limit', type=int, default=0)
    args = p.parse_args()

    out_path = args.out or args.infile.replace('.jsonl', '-labelled.jsonl')
    log_path = args.log or args.infile.replace('.jsonl', '-annotation-log.jsonl')

    key = os.environ.get('OPENROUTER_API_KEY', '')
    if not key:
        raise SystemExit('OPENROUTER_API_KEY is not set. Set it in your shell; do not '
                         'put it in this file.')
    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key=key)

    cases = {}
    for split in ('train', 'valid', 'test'):
        for c in load_dataset('csa')[split]:
            cases[c['uid']] = c

    convs = [json.loads(l) for l in open(args.infile, encoding='utf-8')]
    if args.limit:
        convs = convs[:args.limit]

    # Resume on conversation index, so a rate-limit stop does not repeat paid calls.
    done = 0
    if os.path.exists(out_path):
        with open(out_path, encoding='utf-8') as f:
            done = sum(1 for _ in f)
        if done:
            print('resuming: %d conversations already annotated' % done, flush=True)

    n_calls = n_parsed = 0
    counts = {l: 0 for l in LABELS}
    logf = open(log_path, 'a', encoding='utf-8')
    with open(out_path, 'a' if done else 'w', encoding='utf-8') as outf:
        for ci, conv in enumerate(convs):
            if ci < done:
                continue
            case = cases.get(conv['uid'], {})
            fact_text = {k: v['text'] for k, v in (case.get('private_facts') or {}).items()}
            disclosed_so_far = []   # decisive facts revealed so far
            prior_any = []          # every prior participant utterance
            dialog = conv['dialog']

            for ti, turn in enumerate(dialog):
                # track what has been disclosed before this turn, for the SHARE check
                if turn['speaker'] == 'usr':
                    prior_any.append(turn['content'])
                    for fid in conv.get('revealed', []):
                        if fid in fact_text and fact_text[fid][:40] in turn['content']:
                            disclosed_so_far.append(fact_text[fid])
                if not turn.get('annotate'):
                    continue

                ctx = '\n'.join('%s: %s' % (t['role'], t['content'])
                                for t in dialog[max(0, ti - args.context_turns):ti])
                prompt = TEMPLATE.format(fewshot=FEWSHOT, context=ctx[-2500:],
                                         chair=turn['role'],
                                         utterance=turn['content'][:800])

                raw, err, latency = '', None, 0.0
                for attempt in range(args.max_retries):
                    t0 = time.time()
                    try:
                        r = client.chat.completions.create(
                            model=args.model, temperature=0, max_tokens=args.max_tokens,
                            messages=[{'role': 'system', 'content': SYSTEM},
                                      {'role': 'user', 'content': prompt}])
                        raw = (r.choices[0].message.content or '').strip()
                        latency = time.time() - t0
                        break
                    except Exception as e:
                        err = repr(e)[:200]
                        latency = time.time() - t0
                        # exponential backoff: free tiers rate-limit aggressively
                        time.sleep(min(60, 2 ** attempt))

                label = parse_label(raw)
                share_ok, share_pool = (check_share(turn['content'], prior_any,
                                                    disclosed_so_far)
                                        if label == 'share' else (None, None))
                n_calls += 1
                if label:
                    n_parsed += 1
                    counts[label] += 1
                turn['strategy'] = label
                turn['annotator_raw'] = raw

                rec = {
                    'ts': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'conv_index': ci, 'uid': conv['uid'], 'turn_index': ti,
                    'chair': turn['role'],
                    'utterance': turn['content'][:400],
                    'context_tail': ctx[-300:],
                    'raw_response': raw, 'label': label, 'error': err,
                    'latency_s': round(latency, 2),
                    # mechanical cross-checks: partial ground truth, no human needed
                    'check_decide_consistent':
                        (check_decide(turn['content']) if label == 'decide' else None),
                    'check_share_label_ok': (share_ok if label == 'share' else None),
                    'check_share_pools_decisive': (share_pool if label == 'share' else None),
                }
                logf.write(json.dumps(rec, ensure_ascii=False) + '\n')
                logf.flush()

                if args.sleep:
                    time.sleep(args.sleep)

            outf.write(json.dumps(conv, ensure_ascii=False) + '\n')
            outf.flush()
            print('[annotate] %d/%d %-28s calls=%d parsed=%d  %s'
                  % (ci + 1, len(convs), conv['uid'], n_calls, n_parsed,
                     ' '.join('%s:%d' % (l, counts[l]) for l in LABELS)), flush=True)

    logf.close()
    print('\n--- annotation summary ---')
    print('calls            : %d' % n_calls)
    print('parsed to a label: %d (%.1f%%)' % (n_parsed, 100 * n_parsed / max(1, n_calls)))
    for l in LABELS:
        print('  %-9s %4d  %5.1f%%' % (l, counts[l], 100 * counts[l] / max(1, n_parsed)))
    print('\nwrote %s' % out_path)
    print('log   %s' % log_path)


if __name__ == '__main__':
    main()
