
"""Filtered behaviour cloning: manufacture supervised data for the planner.

CSA ships no dialogues, so PPDPP's supervised stage has nothing to learn from. This
produces the equivalent using the one asset PPDPP's own datasets lack -- executable
ground truth:

  1. roll out k dialogues per scenario with acts sampled UNIFORMLY AT RANDOM. Because
     we choose the acts, every chair turn already carries its label: no annotation
     pass, no LLM labeller, no human agreement study.
  2. score each rollout deterministically on the content and provenance checks.
  3. rank WITHIN each scenario and keep the top few. Ranking within, not globally, so
     hard scenarios still contribute rather than the filter selecting only easy ones.
  4. emit (prefix, act) pairs from the survivors.


The signal comes from the filter, not from a judge or a script. Rollouts cost nothing
beyond the utterances themselves: no critic, and settlement extraction only if the
chair never emitted JSON.

    python make_sft_data.py --split train --k 6 --keep 2 --out data_sft
"""
import argparse
import collections
import io
import json
import os
import random
import statistics as st
import sys

import torch

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from env import Env                                          # noqa: E402
from prompt import CSAAct                                    # noqa: E402
from utils import load_dataset, set_random_seed              # noqa: E402
_CSA_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CSA_ROOT not in sys.path:
    sys.path.insert(0, _CSA_ROOT)
from csa_core.verifier import score                        # noqa: E402

ACTS = sorted(CSAAct.keys())


def build_args(cli):
    class A:
        pass
    a = A()
    a.data_name = 'csa'
    a.system = a.user = a.critic = cli.backend
    a.qwen_path, a.qwen_dtype, a.qwen_device_map = cli.qwen_path, cli.qwen_dtype, cli.device_map
    a.openai_model = cli.openai_model
    a.max_turn, a.max_new_tokens, a.max_seq_length = 36, cli.max_new_tokens, 512
    a.seed, a.gamma = cli.seed, 0.999
    # verifier reward with no judge: rollouts cost only the utterances.
    a.csa_reward = 'verifier'
    a.csa_w_content, a.csa_w_prov = 0.5, 0.5
    a.csa_w_halluc = a.csa_w_schema = a.csa_w_accept = 0.0
    a.csa_w_shape_disc = a.csa_w_shape_elic = a.csa_w_shape_cover = 0.0
    a.critic_every, a.critic_batch = 1, 10
    a.csa_reveal_threshold = cli.reveal_threshold
    a.csa_settlement_max_tokens = 512
    a.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    a.num_gpus, a.max_gpu_memory = 1, None
    a.load_8bit = a.cpu_offloading = a.debug = False
    a.model_path = None
    return a


def rollout(env, rng, chair_turns, force_finalise=True):
    """One dialogue with uniformly random acts. Returns (turns, acts, score, ...).

    The last chair turn is forced to `finalise` unless the policy already chose it.
    Without that, a uniformly random planner selects finalise in only about a third of
    short rollouts, most dialogues end with no settlement at all, and every rollout
    scores zero -- which makes the ranking filter vacuous for a reason that has nothing
    to do with the quality of the dialogue.
    """
    env.reset()
    dm_name = env.names[env.dm]
    acts = {}
    budget = min(chair_turns, env.max_turn)
    for t in range(budget):
        last = (t == budget - 1)
        action = 'decide' if (last and force_finalise) else rng.choice(ACTS)
        before = len(env.conversation)
        env.step(action)
        if len(env.conversation) <= before:
            break
        # the chair turn produced by this action is the last chair utterance
        for i in range(len(env.conversation) - 1, -1, -1):
            if env.conversation[i]['role'] == dm_name:
                acts[i] = action
                break
        if env.cur_conver_step >= env.max_turn:
            break

    # Score whatever settlement exists; fall back to extraction only if the chair never
    # produced parseable JSON, so a rollout is never scored against an empty object
    # merely because it stopped early.
    if not env.settlement:
        env.settlement = env._csa_extract_settlement()
    s = score(env.case, env.settlement, env.revealed)
    return list(env.conversation), acts, s, dict(env.reveal_turn), dict(env.reveal_elicited)


def to_dialog(turns, acts, dm_name):
    out = []
    for i, t in enumerate(turns):
        e = {'role': t['role'], 'content': t['content'],
             'speaker': 'sys' if t['role'] == dm_name else
                        ('env' if t['role'] == 'Meeting' else 'usr')}
        if i in acts:
            e['strategy'] = acts[i]
        out.append(e)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--split', default='train', choices=['train', 'valid', 'test'])
    p.add_argument('--k', type=int, default=6, help='rollouts per scenario')
    p.add_argument('--keep', type=int, default=2, help='top-N per scenario to train on')
    p.add_argument('--chair_turns', type=int, default=5)
    p.add_argument('--out', default='data_sft')
    p.add_argument('--limit', type=int, default=0, help='0 = all scenarios')
    p.add_argument('--backend', default='qwen')
    p.add_argument('--qwen_path', default='Qwen/Qwen2.5-7B-Instruct')
    p.add_argument('--qwen_dtype', default='bfloat16')
    p.add_argument('--device_map', default='cuda:0')
    p.add_argument('--openai_model', default='gpt-3.5-turbo-0613')
    p.add_argument('--max_new_tokens', type=int, default=96)
    p.add_argument('--reveal_threshold', type=float, default=0.35)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--restart', action='store_true',
                   help='Discard any existing output and start from scratch.')
    cli = p.parse_args()

    args = build_args(cli)
    set_random_seed(cli.seed)
    rng = random.Random(cli.seed)
    rows = load_dataset('csa')[cli.split]
    if cli.limit:
        rows = rows[:cli.limit]

    env = Env(args, {'train': rows, 'test': rows}, mode='train')
    env.mode = 'test'          # sequential, and temperature 0 for generation
    env.dataset = rows
    os.makedirs(cli.out, exist_ok=True)
    path = os.path.join(cli.out, 'csa-%s.txt' % cli.split)
    stats_path = os.path.join(cli.out, 'rollout_stats_%s.json' % cli.split)

    # Resume: a full run is several hours, so a crash or a lost session must not throw
    # away completed scenarios. Every finished scenario is already flushed to disk with
    # its uid, so anything present is skipped and the file is appended to.
    done_uids, all_stats = set(), []
    if not cli.restart and os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            for line in f:
                try:
                    done_uids.add(json.loads(line)['uid'])
                except Exception:
                    pass
        if os.path.exists(stats_path):
            try:
                all_stats = json.load(open(stats_path, encoding='utf-8'))
            except Exception:
                all_stats = []
        if done_uids:
            print('resuming: %d scenarios already complete, skipping them'
                  % len(done_uids), flush=True)

    spreads = [s['spread'] for s in all_stats]
    kept_pairs = 0
    mode = 'w' if (cli.restart or not done_uids) else 'a'
    with open(path, mode, encoding='utf-8') as f:
        for i in range(len(rows)):
            if rows[i]['uid'] in done_uids:
                continue
            cands = []
            for _ in range(cli.k):
                env.test_num = i
                turns, acts, s, rt, re_ = rollout(env, rng, cli.chair_turns)
                cands.append({'turns': turns, 'acts': acts,
                              'score': 0.5 * s['cbar'] + 0.5 * s['pbar'],
                              'cbar': s['cbar'], 'pbar': s['pbar'],
                              'disclosure': s['disclosure_rate'],
                              'elicited': sum(1 for v in re_.values() if v),
                              # stored so the score can be recomputed offline: without
                              # the settlement and the revealed set, no reported number
                              # is auditable from the saved transcripts alone.
                              'settlement': dict(env.settlement or {}),
                              'revealed': sorted(env.revealed),
                              'reveal_elicited': dict(re_),
                              'from_chair_json': bool(env.settlement),
                              'dm_name': env.names[env.dm], 'uid': env.case.get('uid')})
            scores = [c['score'] for c in cands]
            spread = max(scores) - min(scores)
            spreads.append(spread)
            all_stats.append({'uid': cands[0]['uid'], 'scores': scores, 'spread': spread,
                              'disclosure': [c['disclosure'] for c in cands]})

            cands.sort(key=lambda c: -c['score'])
            for c in cands[:cli.keep]:
                dialog = to_dialog(c['turns'], c['acts'], c['dm_name'])
                kept_pairs += sum(1 for t in dialog if t.get('strategy'))
                f.write(json.dumps({'uid': c['uid'], 'score': c['score'],
                                    'cbar': c['cbar'], 'pbar': c['pbar'],
                                    'settlement': c['settlement'],
                                    'revealed': c['revealed'],
                                    'reveal_elicited': c['reveal_elicited'],
                                    'from_chair_json': c['from_chair_json'],
                                    'dialog': dialog}, ensure_ascii=False) + '\n')
            f.flush()
            # Flush stats per scenario as well, so a resume keeps the spread history.
            with open(stats_path, 'w', encoding='utf-8') as sf:
                json.dump(all_stats, sf, indent=1)
            print('[%s] %d/%d %-28s scores %s spread %.3f  pairs=%d'
                  % (cli.split, i + 1, len(rows), cands[0]['uid'],
                     ' '.join('%.2f' % x for x in sorted(scores, reverse=True)),
                     spread, kept_pairs), flush=True)

    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(all_stats, f, indent=1)

    nz = [s for s in spreads if s > 1e-9]
    print('\n--- filter viability ---')
    print('scenarios                : %d' % len(spreads))
    print('with non-zero spread     : %d (%.0f%%)' % (len(nz), 100 * len(nz) / max(1, len(spreads))))
    print('mean spread              : %.3f' % (st.mean(spreads) if spreads else 0))
    print('median spread            : %.3f' % (st.median(spreads) if spreads else 0))
    print('labelled pairs kept      : %d' % kept_pairs)
    if len(nz) < 0.5 * len(spreads):
        print('\nWARNING: most scenarios show no score variation across rollouts, so '
              'ranking is selecting noise. The filter has little signal to work with.')
    print('\nwrote %s and %s' % (path, stats_path))


if __name__ == '__main__':
    main()
