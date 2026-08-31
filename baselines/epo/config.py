"""Paths, dataset loading and every default in one place.

This package sits BESIDE ppdpp_csa/ and never modifies it. What it reuses from there
is deliberately limited to the two things that must be byte-identical for the two
planners to be comparable at all:

  * the scenario splits in ppdpp_csa/data/csa-{train,valid,test}.txt
  * verifier.score / floor_score

Everything else -- prompts, turn loop, reward, policy -- is reimplemented here, because
that is exactly what the EPO port changes.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _find_ppdpp_csa():
    """Locate ppdpp_csa/ wherever this package has been moved to.

    A hardcoded relative path breaks the moment the folder is relocated, so the
    directory is discovered instead: an explicit override first, then a walk up the tree
    looking for the files this package actually reads. Fails with a usable message
    rather than a ModuleNotFoundError three imports later.
    """
    def ok(d):
        return (d and os.path.isfile(os.path.join(d, 'verifier.py'))
                and os.path.isfile(os.path.join(d, 'prompt.py'))
                and os.path.isdir(os.path.join(d, 'data')))

    env = os.environ.get('CSA_PPDPP_DIR')
    if env:
        if not ok(os.path.abspath(env)):
            raise SystemExit('CSA_PPDPP_DIR=%r does not look like ppdpp_csa/ '
                             '(needs verifier.py, prompt.py and data/)' % env)
        return os.path.abspath(env)

    seen, node = [], HERE
    for _ in range(6):                       # this dir, then five ancestors
        for cand in (os.path.join(node, 'ppdpp_csa'),
                     os.path.join(node, 'ppdpp', 'ppdpp_csa'),
                     os.path.join(node, 'baselines', 'ppdpp', 'ppdpp_csa')):
            seen.append(os.path.abspath(cand))
            if ok(cand):
                return os.path.abspath(cand)
        parent = os.path.dirname(node)
        if parent == node:
            break
        node = parent

    raise SystemExit(
        'cannot find ppdpp_csa/. This package reads its splits and verifier from there.\n'
        'Looked in:\n  %s\n\n'
        'Fix: point CSA_PPDPP_DIR at it, e.g.\n'
        '  export CSA_PPDPP_DIR=/path/to/baselines/ppdpp/ppdpp_csa'
        % '\n  '.join(dict.fromkeys(seen)))


PPDPP = _find_ppdpp_csa()
# raw/ sits beside ppdpp_csa/, not beside this package.
RAW = os.path.abspath(os.path.join(PPDPP, '..', 'raw'))

DATA = os.path.join(HERE, 'data')
LOGS = os.path.join(HERE, 'logs')
CKPT = os.path.join(HERE, 'ckpt')
for _d in (DATA, LOGS, CKPT):
    os.makedirs(_d, exist_ok=True)

# ppdpp_csa is a flat package (its modules import each other by bare name), so it has
# to go on sys.path rather than be imported as a subpackage.
if PPDPP not in sys.path:
    sys.path.insert(0, PPDPP)


# ---------------------------------------------------------------- dataset
def load_csa(split=None):
    """The exact splits the PPDPP runs used: 99 / 9 / 42, scenario-disjoint.

    ppdpp_csa/utils.load_dataset resolves './data' relative to the CURRENT working
    directory, so calling it from here would silently look in epo/data and find
    nothing. The paths are pinned instead.
    """
    out = {}
    for key in ('train', 'valid', 'test'):
        path = os.path.join(PPDPP, 'data', 'csa-%s.txt' % key)
        rows = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip('\n')
                if line:
                    rows.append(eval(line))     # same format load_dataset expects
        out[key] = rows
    if split:
        return out[split]
    return out


def case_index():
    """uid -> case, across all splits."""
    idx = {}
    for rows in load_csa().values():
        for r in rows:
            idx[r['uid']] = r
    return idx


# ---------------------------------------------------------------- defaults
class Defaults:
    """Every knob, with the value chosen for THIS budget rather than EPO's.

    Where a value departs from the paper the reason is on the line. See
    epo-vs-vanilla.pdf for the full argument.
    """
    # --- dialogue agent (frozen). EPO uses Llama3-8B / GPT-4o; we hold this fixed to
    # the same backend the PPDPP runs used, so LLM_d is constant across the comparison.
    agent_model = 'Qwen/Qwen2.5-7B-Instruct'
    agent_dtype = 'bfloat16'
    agent_device = 'cuda:0'
    agent_max_new_tokens = 96
    settlement_max_tokens = 512          # a settlement JSON truncates at 96
    agent_temperature = 0.7              # train; eval forces greedy

    # --- strategist (trained)
    strategist_model = 'Qwen/Qwen2.5-7B-Instruct'
    strategist_device = 'cuda:1'         # set to cuda:0 to co-reside; see README
    lora_r = 16                          # EPO full-FTs; 99 scenarios would memorise
    lora_alpha = 32
    lora_dropout = 0.05
    lora_targets = ('q_proj', 'k_proj', 'v_proj', 'o_proj',
                    'gate_proj', 'up_proj', 'down_proj')
    strategy_max_tokens = 48             # ~20 words plus the act tag
    strategy_temperature = 0.7
    tag_weight = 2.0                     # the act tag is ~2 of ~15 tokens

    # --- RL. lr is 30x EPO's 1e-6: that is a full-FT lr and barely moves LoRA.
    lr = 3e-5
    warmup_frac = 0.05
    gamma = 0.99
    episodes_per_update = 4              # 700 episodes -> ~175 optimizer steps
    grad_clip = 1.0
    kl_beta = 0.01                       # EPO reports none; insurance against collapse
    total_episodes = 700
    group_k = 4                          # rollouts per scenario, for the baseline

    # --- reward
    prm = 'verifier'                     # 'verifier' | 'judge'
    prm_mode = 'binary'                  # 'binary' (EPO-faithful) | 'graded'
    done_tau = 0.6                       # dca threshold for success
    reveal_threshold = 0.35              # FROZEN. Changing it invalidates every
                                         # disclosure figure already reported.
    leak_invalidates = True
    w_use, w_pool, w_close = 0.5, 0.3, 0.2
    w_halluc_pen = 0.5
    resolve_provenance = True

    # --- manufacturing (stage 1)
    or_base_url = 'https://openrouter.ai/api/v1'
    or_model = 'google/gemma-3-27b-it'   # override with --model
    or_max_tokens = 48
    or_max_retries = 6

    seed = 1
