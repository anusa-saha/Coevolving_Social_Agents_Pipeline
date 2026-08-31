"""STAGE 1b -- turn collected episodes into reward-model training data.

One row per chair turn:

    prompt      the chair's view + transcript up to that turn   (state s_t)
    completion  the utterance that was actually said            (action a_t)
    label       the attributed scalar r_t

The label is where this departs from the paper. Sotopia-RL calls GPT-4o with the full
episode in view and asks how much each utterance contributed. CSA states the answer:
decisive_facts says which checks each private fact controls, and the transcript says
which turn drew it out. No model, no API, byte-for-byte reproducible.

Normalisation is fitted here, across the whole dataset, and saved alongside the data so
the reward model and everything downstream share one scale.

    python make_rm_data.py --episodes data/episodes-train.jsonl
"""
import argparse
import json
import os
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding='utf-8', errors='replace')
    except Exception:                                # noqa: BLE001
        pass

import attribution                                   # noqa: E402
import config                                        # noqa: E402
import data_csa                                      # noqa: E402
import paths                                         # noqa: E402
import prompts_sr as P                               # noqa: E402


def chair_turn_indices(dialog):
    return [i for i, t in enumerate(dialog) if t.get('speaker') == 'sys']


def acts_from_dialog(dialog):
    """If the corpus carries planner act labels (an annotated set from another baseline),
    use them; otherwise return None and let the lexical detector decide."""
    acts, step = {}, -1
    for t in dialog:
        if t.get('speaker') == 'sys':
            step += 1
            if t.get('strategy'):
                acts[step] = t['strategy']
    return acts or None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--episodes', nargs='+',
                   default=[os.path.join(paths.DATA, 'episodes-train.jsonl')])
    p.add_argument('--out', default=os.path.join(paths.DATA, 'rm-train.jsonl'))
    p.add_argument('--norm_out', default=os.path.join(paths.DATA, 'normaliser.json'))
    p.add_argument('--within_episode_norm', action='store_true',
                   help='reproduce the reading of Eq.4 in which G cancels; ablation only')
    p.add_argument('--use_annotated_acts', action='store_true',
                   help='trust planner act labels if present, instead of is_eliciting')
    p.add_argument('--weights', default='',
                   help='comma-separated pool,use,cover weights (default: equal)')
    cli = p.parse_args()

    cases = data_csa.case_index()
    eps = []
    for path in cli.episodes:
        if not os.path.exists(path):
            raise SystemExit('missing %s -- run collect_episodes.py first' % path)
        eps += [json.loads(l) for l in open(path, encoding='utf-8')]
    print('episodes: %d' % len(eps))

    weights = None
    if cli.weights:
        vals = [float(x) for x in cli.weights.split(',')]
        weights = dict(zip(attribution.DIMS, vals))

    raws, keep = [], []
    dropped = 0
    for e in eps:
        case = cases.get(e['uid'])
        if not case:
            dropped += 1
            continue
        acts = acts_from_dialog(e['dialog']) if cli.use_annotated_acts else None
        raw, diag = attribution.attribute(
            case, e['dialog'], e.get('settlement'), acts=acts,
            gate_on_integrity=config.Defaults.gate_on_integrity)
        raws.append(raw)
        keep.append((e, case, raw, diag))

    norm = attribution.Normaliser(weights=weights,
                                  within_episode=cli.within_episode_norm).fit(raws)

    n_rows, gated = 0, 0
    with open(cli.out, 'w', encoding='utf-8') as f:
        for e, case, raw, diag in keep:
            labels = norm.apply(raw)
            if not diag['gate']:
                gated += 1
            idxs = chair_turn_indices(e['dialog'])
            for step, di in enumerate(idxs):
                if step >= len(labels):
                    break
                prefix = [{'role': t['role'], 'content': t['content']}
                          for t in e['dialog'][:di]]
                msgs = P.chair_messages(case, prefix, settling=False)
                f.write(json.dumps({
                    'uid': e['uid'], 'turn': step,
                    'prompt_messages': msgs,
                    'completion': e['dialog'][di]['content'],
                    'label': labels[step],
                    'raw': {d: raw[d][step] for d in attribution.DIMS},
                    'gate': diag['gate'],
                }, ensure_ascii=False) + '\n')
                n_rows += 1

    with open(cli.norm_out, 'w', encoding='utf-8') as f:
        json.dump(norm.state(), f, indent=1)

    vals = [x for _e, _c, raw, _d in keep for x in norm.apply(raw)]
    print('wrote %s  (%d rows, %d episodes gated to zero, %d unknown uid)'
          % (cli.out, n_rows, gated, dropped))
    print('label distribution: %s' % attribution.describe(vals))
    for d in attribution.DIMS:
        v = [x for _e, _c, raw, _d in keep for x in raw[d]]
        print('  %-6s raw %s' % (d, attribution.describe(v)))
    print('normaliser -> %s' % cli.norm_out)
    if attribution.describe(vals).get('frac_non_zero', 0) < 0.15:
        print('\nWARNING: fewer than 15%% of turns carry a non-zero label. The reward '
              'model will collapse toward predicting 0. Collect more episodes, or check '
              'that eliciting turns are being detected.')


if __name__ == '__main__':
    main()
