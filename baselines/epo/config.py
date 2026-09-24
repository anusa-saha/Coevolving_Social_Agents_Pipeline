"""Paths, dataset loading and every default in one place.

The split, the disclosure detector and the verifier come from csa_core, so every arm
is measured with one instrument.

One thing is still borrowed from ppdpp_csa, deliberately: prompt.py, the LLM_d prompt
layer. Holding the dialogue agent's prompt byte-identical across PPDPP and EPO is what
makes the two arms comparable at all -- reimplementing it here would quietly change the
experiment rather than tidy it. Everything else (turn loop, reward, policy) is EPO's own,
because that is exactly what the port changes.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(HERE)
if _ROOT not in sys.path:                    # so csa_core imports without an install
    sys.path.insert(0, _ROOT)


def _find_ppdpp_csa():
    """Locate ppdpp_csa/ wherever this package has been moved to.

    A hardcoded relative path breaks the moment the folder is relocated, so the
    directory is discovered instead: an explicit override first, then a walk up the tree
    looking for the files this package actually reads. Fails with a usable message
    rather than a ModuleNotFoundError three imports later.
    """
    def ok(d):
        # Only prompt.py now. The split and the verifier moved to csa_core; the prompt
        # layer stays borrowed on purpose, so LLM_d is byte-identical across PPDPP and
        # EPO and the two arms remain comparable.
        return d and os.path.isfile(os.path.join(d, 'prompt.py'))

    env = os.environ.get('CSA_PPDPP_DIR')
    if env:
        if not ok(os.path.abspath(env)):
            raise SystemExit('CSA_PPDPP_DIR=%r does not look like ppdpp_csa/ '
                             '(needs prompt.py)' % env)
        return os.path.abspath(env)

    seen, node = [], HERE
    for _ in range(6):                       # this dir, then five ancestors
        for cand in (os.path.join(node, 'ppdpp'),
                     os.path.join(node, 'baselines', 'ppdpp')):
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
        '  export CSA_PPDPP_DIR=/path/to/baselines/ppdpp'
        % '\n  '.join(dict.fromkeys(seen)))


PPDPP = _find_ppdpp_csa()

from csa_core import data_csa, paths  # noqa: E402
# raw/ sits beside ppdpp_csa/, not beside this package.
RAW = paths.find_raw()          # the shared dataset, one copy at the repo root

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
    """The scenario split, from csa_core.

    Re-derived by csa_core.data_csa rather than read out of ppdpp/data/csa-*.txt, so this
    arm follows the configured benchmark (domain list and per-domain cap) instead of a
    file pinned to one old configuration.
    """
    if split:
        return data_csa.load(split)
    # Every split the loader has, not a hardcoded three: with the explicit split files
    # that includes the test_seen / test_unseen views, and case_index() below needs
    # them or the unseen-domain scenarios go missing from the index.
    return dict(data_csa.load())


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
    agent_model = '/scratch/rohank__iitp/Qwen3-8B'
    agent_dtype = 'bfloat16'
    agent_device = 'cuda:0'
    agent_max_new_tokens = 96
    settlement_max_tokens = 512          # a settlement JSON truncates at 96
    agent_temperature = 0.7              # train; eval forces greedy

    # --- strategist (trained)
    strategist_model = '/scratch/rohank__iitp/Qwen3-8B'
    strategist_device = 'cuda:1'         # set to cuda:0 to co-reside; see README
    lora_r = 16                          # EPO full-FTs; 99 scenarios would memorise
    lora_alpha = 32
    lora_dropout = 0.05
    # q/k/v/o exist only in Qwen3.5's full-attention layers (every 4th). The other 24 are
    # Gated DeltaNet, projected by in_proj_qkv / in_proj_z / out_proj -- without those,
    # three attention blocks in four would carry no adapter.
    lora_targets = ('q_proj', 'k_proj', 'v_proj', 'o_proj',
                    'in_proj_qkv', 'in_proj_z', 'out_proj',
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
    grad_checkpointing = False           # ~6x less activation memory, ~30% slower
    max_prompt_tokens = 1536             # matches the other arms; 0 disables the cap
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
    or_base_url = paths.ANNOTATOR_BASE_URL
    or_model = paths.ANNOTATOR_MODEL     # shared with PPDPP; --model overrides
    or_max_tokens = 48
    or_max_retries = 6

    seed = 1
