"""Generate and score meeting transcripts, to be labelled afterwards by an annotator.

The chair speaks UNPROMPTED -- no act conditioning. That is essential: if we told the
chair which intent to perform, the annotator would only recover our own instruction and
the classifier would learn the prompt rather than dialogue structure.

For each scenario, k conversations are generated and scored deterministically on the
dataset's content and provenance checks. The top few *within that scenario* are kept, so
hard scenarios still contribute rather than the filter selecting only easy ones.

Output carries the transcript, the settlement, the score and the disclosure record, but
NO labels. annotate_intents.py adds those.

    python generate_conversations.py --split train --k 5 --keep 3 --out conversations
"""
import argparse
import json
import os
import statistics as st
import sys

import torch

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from env import Env                                          # noqa: E402
from utils import load_dataset, set_random_seed              # noqa: E402
_CSA_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CSA_ROOT not in sys.path:
    sys.path.insert(0, _CSA_ROOT)
from csa_core.verifier import score               # noqa: E402


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
    # verifier reward, no judge: a rollout costs only its utterances.
    a.csa_reward = 'verifier'
    a.csa_w_content, a.csa_w_prov = 0.5, 0.5
    a.csa_w_halluc = a.csa_w_schema = a.csa_w_accept = 0.0
    a.csa_w_shape_disc = a.csa_w_shape_elic = a.csa_w_shape_cover = 0.0
    a.critic_every, a.critic_batch = 1, 10
    a.csa_reveal_threshold = cli.reveal_threshold
    a.csa_settlement_max_tokens = cli.settlement_tokens
    a.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    a.num_gpus, a.max_gpu_memory = 1, None
    a.load_8bit = a.cpu_offloading = a.debug = False
    a.model_path = None
    return a


def rollout(env, chair_turns):
    """One unprompted conversation, then a settlement turn so it can be scored.

    Only the unprompted turns are labelled later; the closing settlement turn is marked
    so the annotator skips it -- it was instructed, so its intent is known and including
    it would inflate the DECIDE class.
    """
    env.reset()
    dm_name = env.names[env.dm]
    budget = min(chair_turns, env.max_turn)

    natural = max(1, budget - 1)
    for _ in range(natural):
        before = len(env.conversation)
        env.step(None)                      # unprompted: no act instruction
        if len(env.conversation) <= before or env.cur_conver_step >= env.max_turn:
            break

    n_natural = len(env.conversation)
    if env.cur_conver_step < env.max_turn:
        env.step('decide')                  # instructed: produces the settlement JSON

    if not env.settlement:
        env.settlement = env._csa_extract_settlement()
    s = score(env.case, env.settlement, env.revealed)
    return env, list(env.conversation), n_natural, s


def to_dialog(turns, n_natural, dm_name):
    out = []
    for i, t in enumerate(turns):
        out.append({'role': t['role'], 'content': t['content'],
                    'speaker': 'sys' if t['role'] == dm_name else
                               ('env' if t['role'] == 'Meeting' else 'usr'),
                    # instructed turns are excluded from annotation
                    'annotate': t['role'] == dm_name and i < n_natural})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--split', default='train', choices=['train', 'valid', 'test'])
    p.add_argument('--k', type=int, default=5, help='conversations per scenario')
    p.add_argument('--keep', type=int, default=3, help='top-N per scenario to keep')
    p.add_argument('--chair_turns', type=int, default=5)
    p.add_argument('--out', default='conversations')
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--backend', default='qwen')
    p.add_argument('--qwen_path', default='Qwen/Qwen2.5-7B-Instruct')
    p.add_argument('--qwen_dtype', default='bfloat16')
    p.add_argument('--device_map', default='cuda:0')
    p.add_argument('--openai_model', default='gpt-3.5-turbo-0613')
    p.add_argument('--max_new_tokens', type=int, default=96)
    p.add_argument('--settlement_tokens', type=int, default=512)
    p.add_argument('--reveal_threshold', type=float, default=0.35)
    p.add_argument('--temperature', type=float, default=0.8,
                   help='Sampling temperature during generation. Must be > 0 or the k '
                        'rollouts of a scenario are identical and ranking is vacuous.')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--restart', action='store_true')
    p.add_argument('--uids_from', default='',
                   help='Restrict generation to the scenario uids appearing in this '
                        'jsonl (e.g. data_sft/csa-train.txt). Augmenting the training '
                        'set must never add conversations for a scenario that sits in '
                        'the SFT test or validation split, or the held-out numbers stop '
                        'being held out.')
    cli = p.parse_args()

    args = build_args(cli)
    set_random_seed(cli.seed)
    rows = load_dataset('csa')[cli.split]
    if cli.uids_from:
        keep = set()
        with open(cli.uids_from, encoding='utf-8') as f:
            for line in f:
                try:
                    keep.add(json.loads(line)['uid'])
                except Exception:
                    pass
        before = len(rows)
        rows = [r for r in rows if r['uid'] in keep]
        print('uid filter: %d of %d %s scenarios are in %s'
              % (len(rows), before, cli.split, cli.uids_from), flush=True)
    if cli.limit:
        rows = rows[:cli.limit]
    if not rows:
        print('no scenarios to generate for split=%s after filtering' % cli.split)
        return

    env = Env(args, {'train': rows, 'test': rows}, mode='train')
    env.mode = 'test'                       # sequential iteration over the split
    env.force_temperature = cli.temperature  # but sample, or every rollout is identical
    env.dataset = rows
    os.makedirs(cli.out, exist_ok=True)
    path = os.path.join(cli.out, 'conversations-%s.jsonl' % cli.split)

    # Resume: a full run is hours, so completed scenarios are skipped on restart.
    done = set()
    if not cli.restart and os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            for line in f:
                try:
                    done.add(json.loads(line)['uid'])
                except Exception:
                    pass
        if done:
            print('resuming: %d scenarios already done' % len(done), flush=True)

    spreads, kept, chair_turns_total, from_json = [], 0, 0, 0
    mode = 'w' if (cli.restart or not done) else 'a'
    with open(path, mode, encoding='utf-8') as f:
        for i in range(len(rows)):
            if rows[i]['uid'] in done:
                continue
            cands = []
            for _ in range(cli.k):
                env.test_num = i
                e, turns, n_nat, s = rollout(env, cli.chair_turns)
                cands.append({
                    'uid': e.case.get('uid'), 'domain': e.case.get('domain'),
                    'num_agents': e.case.get('num_agents'),
                    'score': 0.5 * s['cbar'] + 0.5 * s['pbar'],
                    'cbar': s['cbar'], 'pbar': s['pbar'],
                    'disclosure': s['disclosure_rate'],
                    'settlement': dict(e.settlement or {}),
                    'revealed': sorted(e.revealed),
                    'reveal_elicited': dict(e.reveal_elicited),
                    'leaks': list(e.leaks),
                    'dialog': to_dialog(turns, n_nat, e.names[e.dm])})
            scores = [c['score'] for c in cands]
            spreads.append(max(scores) - min(scores))
            cands.sort(key=lambda c: -c['score'])
            for c in cands[:cli.keep]:
                n_ann = sum(1 for t in c['dialog'] if t['annotate'])
                chair_turns_total += n_ann
                kept += 1
                from_json += bool(c['settlement'])
                f.write(json.dumps(c, ensure_ascii=False) + '\n')
            f.flush()
            print('[%s] %d/%d %-28s scores %s spread %.3f  kept=%d turns=%d'
                  % (cli.split, i + 1, len(rows), cands[0]['uid'],
                     ' '.join('%.2f' % x for x in sorted(scores, reverse=True)),
                     spreads[-1], kept, chair_turns_total), flush=True)

    nz = [s for s in spreads if s > 1e-9]
    print('\n--- generation summary ---')
    print('scenarios              : %d' % len(spreads))
    print('non-zero spread        : %d (%.0f%%)'
          % (len(nz), 100 * len(nz) / max(1, len(spreads))))
    print('mean spread            : %.3f' % (st.mean(spreads) if spreads else 0))
    print('conversations kept     : %d' % kept)
    print('chair turns to annotate: %d' % chair_turns_total)
    print('\nwrote %s' % path)


if __name__ == '__main__':
    main()
