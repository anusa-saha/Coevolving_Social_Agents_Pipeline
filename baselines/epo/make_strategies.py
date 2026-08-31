"""STAGE 1 -- manufacture the SFT targets.

ppdpp_csa/data_sft/ already holds 949 labelled chair turns (650 train / 164 valid /
135 test). Each gives the prompt and the ACT TAG, but nothing after the colon: the
annotator was instructed "Answer with exactly one word", so no rationale was ever
recorded. EPO's target is the full "tau: sigma" line, so sigma has to be produced.

The chair's real next utterance is the ground truth of what it did; a small model only
has to compress it into an instruction. Crucially tau is SUPPLIED, not predicted, so
this pass cannot introduce label noise -- the annotated tags survive untouched and only
sigma is synthesised.

Reads OPENROUTER_API_KEY from the environment. Never put a key in this file.

    python make_strategies.py --split train
    python make_strategies.py --split train --fallback_only    # no API at all
"""
import argparse
import collections
import json
import os
import random
import re
import sys
import time

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding='utf-8', errors='replace')
    except Exception:                                # noqa: BLE001
        pass

import config                                        # noqa: E402
import prompt_epo as pe                              # noqa: E402

SFT_DIR = os.path.join(config.PPDPP, 'data_sft')


# ------------------------------------------------------------------ fallback
def template_sigma(tau, utterance, names):
    """Model-free sigma, from the turn's own structure.

    Deterministic and free. Low diversity, so it is a fallback and a control -- if the
    verbalised sigma does not beat this at SFT, the verbaliser added nothing.
    """
    addressed = [n for n in names if n.split()[-1].lower() in utterance.lower()]
    who = addressed[0] if addressed else 'the advisor who has not spoken'
    if tau == 'ask':
        return 'ask %s about a constraint nobody has raised yet' % who
    if tau == 'followup':
        return 'press %s for the exact figure their answer left vague' % who
    if tau == 'share':
        return 'relay what %s disclosed and ask the others how it changes the decision' % who
    return 'record the settlement now as JSON matching the schema'


# ------------------------------------------------------------------ io
def load_labelled(split):
    """(prefix turns, chair name, utterance, tau) for every labelled chair turn."""
    path = os.path.join(SFT_DIR, 'csa-%s.txt' % split)
    if not os.path.exists(path):
        raise SystemExit('missing %s -- run ppdpp_csa/build_sft_splits.py first' % path)
    out = []
    with open(path, encoding='utf-8') as f:
        for ci, line in enumerate(f):
            conv = json.loads(line)
            dialog = conv['dialog']
            for ti, turn in enumerate(dialog):
                tau = turn.get('strategy')
                if not tau:
                    continue
                out.append({
                    'uid': conv['uid'], 'conv_index': ci, 'turn_index': ti,
                    'chair': turn['role'], 'utterance': turn['content'], 'tau': tau,
                    'prefix': [{'role': t['role'], 'content': t['content']}
                               for t in dialog[:ti]],
                })
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--split', default='train', choices=['train', 'valid', 'test'])
    p.add_argument('--out', default='')
    p.add_argument('--log', default='')
    p.add_argument('--model', default=config.Defaults.or_model)
    p.add_argument('--base_url', default=config.Defaults.or_base_url)
    p.add_argument('--context_turns', type=int, default=6)
    p.add_argument('--max_tokens', type=int, default=config.Defaults.or_max_tokens)
    p.add_argument('--max_retries', type=int, default=config.Defaults.or_max_retries)
    p.add_argument('--sleep', type=float, default=0.0,
                   help='fixed pause between calls, for free-tier rate limits')
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--fallback_only', action='store_true',
                   help='no API calls: template sigma for every turn')
    p.add_argument('--seed', type=int, default=0)
    cli = p.parse_args()

    rng = random.Random(cli.seed)
    out_path = cli.out or os.path.join(config.DATA, 'strategies-%s.jsonl' % cli.split)
    log_path = cli.log or os.path.join(config.LOGS, 'make-strategies-%s.jsonl' % cli.split)

    rows = load_labelled(cli.split)
    if cli.limit:
        rows = rows[:cli.limit]
    print('%s: %d labelled chair turns' % (cli.split, len(rows)))

    cases = config.case_index()
    client = None
    if not cli.fallback_only:
        key = os.environ.get('OPENROUTER_API_KEY', '')
        if not key:
            raise SystemExit('OPENROUTER_API_KEY is not set. Set it in your shell; do '
                             'not put it in this file. Or pass --fallback_only.')
        from openai import OpenAI
        client = OpenAI(base_url=cli.base_url, api_key=key)

    # Resume on row count, so a rate-limit stop does not repeat calls already paid for.
    done = 0
    if os.path.exists(out_path):
        with open(out_path, encoding='utf-8') as f:
            done = sum(1 for _ in f)
        if done:
            print('resuming: %d rows already written' % done, flush=True)

    counts = collections.Counter()
    n_api = n_fallback = 0
    logf = open(log_path, 'a', encoding='utf-8')
    with open(out_path, 'a' if done else 'w', encoding='utf-8') as outf:
        for i, r in enumerate(rows):
            if i < done:
                continue
            case = cases.get(r['uid'], {})
            names = [a['name'] for a in (case.get('agents') or [])]
            ctx = '\n'.join('%s: %s' % (t['role'], t['content'])
                            for t in r['prefix'][-cli.context_turns:])

            sigma, source, raw, err = '', 'fallback', '', None
            if client is not None:
                prompt = pe.verbalise_prompt(ctx, r['chair'], r['utterance'], r['tau'])
                for attempt in range(cli.max_retries):
                    try:
                        resp = client.chat.completions.create(
                            model=cli.model, temperature=0.3, max_tokens=cli.max_tokens,
                            messages=[{'role': 'system', 'content': pe.VERBALISE_SYSTEM},
                                      {'role': 'user', 'content': prompt}])
                        raw = (resp.choices[0].message.content or '').strip()
                        cand = pe.clean_sigma(raw)
                        if len(cand.split()) >= 3:
                            sigma, source = cand, 'model'
                            break
                    except Exception as e:           # noqa: BLE001
                        err = repr(e)[:300]
                        time.sleep(min(2 ** attempt, 30))
                if cli.sleep:
                    time.sleep(cli.sleep)

            if not sigma:
                sigma = template_sigma(r['tau'], r['utterance'], names)
                source = 'fallback'
                n_fallback += 1
            else:
                n_api += 1
            counts[r['tau']] += 1

            rec = dict(r)
            rec.update({'sigma': sigma, 'source': source,
                        'target': pe.format_strategy(r['tau'], sigma)})
            outf.write(json.dumps(rec, ensure_ascii=False) + '\n')
            outf.flush()
            logf.write(json.dumps({'i': i, 'uid': r['uid'], 'tau': r['tau'],
                                   'source': source, 'raw': raw[:200],
                                   'sigma': sigma, 'error': err},
                                  ensure_ascii=False) + '\n')
            logf.flush()

            if (i + 1) % 25 == 0:
                print('  %4d/%d  model=%d fallback=%d' % (i + 1, len(rows), n_api,
                                                          n_fallback), flush=True)
    logf.close()

    print('\nwrote %s' % out_path)
    print('  from model    : %d' % n_api)
    print('  from template : %d' % n_fallback)
    print('  by act        : %s' % dict(sorted(counts.items())))
    if counts:
        top, n = counts.most_common(1)[0]
        tot = sum(counts.values())
        print('  imbalance     : "%s" is %.0f%% of turns; the rarest is %.0f%%. '
              'Use --class_balance in sft_epo.py.'
              % (top, 100 * n / tot, 100 * min(counts.values()) / tot))


if __name__ == '__main__':
    main()
