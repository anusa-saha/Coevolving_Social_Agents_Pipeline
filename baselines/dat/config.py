"""Every knob for the DAT port, with the value chosen for THIS budget rather than the
paper's, and the reason on the line where it departs.

DAT (Dialogue Action Tokens, ICLR 2025 submission 8922) freezes the language model and
trains a small planner that emits L continuous prefix-token embeddings before each of the
steered agent's utterances. Two stages: self-cloning (fit planner + up-mapping so the
steered agent reproduces the unsteered one) and offline RL (TD3+BC on a replay buffer
collected by perturbing the self-cloned action).

The CSA mapping, once, because everything else follows from it:

    paper                     CSA
    ------------------------  ------------------------------------------------------
    steered agent Q           the CHAIR (case['decision_maker'])
    partner agent P           the advisors, same frozen Qwen2.5-7B-Instruct
    a round                   one chair turn plus the advisor replies before the next
    judge model reward        the dataset's executable checks, via csa_core.verifier
    Sotopia 7-dimension score dca, disclosure and the settlement gate (see reward_dat)

The reward is the one place CSA is *better* off than the paper: DAT's limitation section
names cheap stable reward signals as the real constraint, and CSA ships decisive_facts,
so criticality is a lookup rather than a judgement. No judge model is called anywhere in
this arm.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import paths                                         # noqa: E402,F401
from csa_core import compat                          # noqa: E402
from csa_core import data_csa                        # noqa: E402

DATA, LOGS, CKPT = paths.DATA, paths.LOGS, paths.CKPT


def load_csa(split=None):
    if split:
        return data_csa.load(split)
    return {k: data_csa.load(k) for k in ('train', 'valid', 'test')}


def case_index():
    return data_csa.case_index()


def preflight():
    """csa_core.compat.report(), minus peft. Returns 1 when something really blocks.

    compat treats a missing peft as blocking because the other arms train an adapter with
    it. DAT trains none -- its only parameters are a two-layer MLP and one matrix, both
    plain torch.nn -- so this arm runs end to end on a box without peft, and refusing to
    start would send someone off to install a dependency it never imports.
    """
    blocking = [x for x in compat.problems() if not x.lower().startswith('peft')]
    compat.report()
    if len(blocking) != len(compat.problems()):
        print()
        print('(peft is listed above; DAT does not import it, so it is not '
              'blocking here)')
    return 1 if blocking else 0


class Defaults:
    # --- dialogue backend (FROZEN, and frozen is the point: DAT never updates an LM
    # parameter). Identical to every other arm so a DAT row is comparable to a PPDPP row.
    model = 'Qwen/Qwen2.5-7B-Instruct'
    dtype = 'bfloat16'
    device = 'cuda:0'
    max_new_tokens = 96
    settlement_max_tokens = 512          # a settlement JSON truncates at 96
    temperature = 0.7                    # rollouts; evaluation forces greedy

    # --- DAT architecture (Section 4 of the paper)
    n_prefix = 2                         # L. The paper's sweep (Fig. 5) peaks at 2 and
                                         # says more tokens degrade language quality.
    action_dim = 64                      # d'. The social-capability experiment's value;
                                         # the red-teaming one used 128.
    planner_hidden = 512
    planner_layers = 2                   # "a small multi-layer perceptron"
    state_layer = -1                     # last transformer layer, last token: g_theta

    # --- stage 1: self-cloning (Section 5.1)
    selfclone_episodes = 120             # the paper collects M dialogues with the
                                         # unsteered policy; 120 x ~3 chair turns is
                                         # ~360 supervised utterances, which is what a
                                         # 2-layer MLP over a frozen encoder needs.
    selfclone_epochs = 3
    selfclone_lr = 1e-4
    selfclone_accum = 8                  # batch 1 x 8: the backward runs THROUGH the
                                         # frozen 7B to reach the prefix, so activation
                                         # memory, not parameter memory, is the limit.
    selfclone_grad_checkpointing = True   # the stage-1 backward runs through the whole
                                         # frozen 7B to reach a 2-token prefix, so
                                         # activation memory is the binding constraint.
                                         # Checkpointing is the right lever; truncation
                                         # is not, because clipping the front of the
                                         # prompt deletes the persona the state encodes.
    selfclone_max_ctx = 0                # 0 = no truncation. Raise off zero only to
                                         # escape an OOM, and expect the clipped
                                         # scenarios to clone worse.
    selfclone_max_target = 96

    # --- stage 2a: replay buffer (Section 6.1)
    buffer_episodes = 400                # the paper collects 10,000 episodes x 3 steps.
                                         # 400 x ~3 = ~1200 transitions. Fig. 4 shows
                                         # ASR still climbing at 80k steps, so this is
                                         # the single most under-resourced number here
                                         # and the first one to raise given GPU time.
    explore_sigma = 0.25                 # N(0, 0.25) in the action space, as the paper
    collect_greedy = True                # "we set the LM agent's temperature to 0 while
                                         # collecting exploration samples for a less
                                         # noisy signal"

    # --- stage 2b: TD3+BC (Section 5.2, Appendix B)
    td3_steps = 4000                     # the paper trains 1 epoch over a 25x larger
                                         # buffer; one pass over 1200 transitions is
                                         # ~5 updates, which trains nothing.
    td3_batch = 256
    td3_lr = 3e-4
    td3_gamma = 0.99
    td3_tau = 0.005
    td3_policy_noise = 0.2               # x max_action
    td3_noise_clip = 0.5                 # x max_action
    td3_policy_freq = 2
    td3_alpha = 2.5                      # BC weight, lambda = alpha / mean|Q|
    td3_reward_weight = 2.0              # weight on transitions carrying a non-zero
                                         # reward in the critic's MSE. Appendix B says
                                         # "a weighted mean squared error loss for
                                         # Q-learning" without giving the weighting;
                                         # this is our reading, and 1.0 disables it.
    residual_rl = True                   # Appendix B: a = pi_phi(s) + pi_phi'((s-mu)/sigma),
                                         # freezing the self-cloned pi_phi so the RL head
                                         # sees normalised input and starts near zero.

    # --- reward (see reward_dat.py). Mirrors EPO's verifier PRM so the two RL arms are
    # supervised by the same deterministic signal; selftest.py asserts they agree.
    reward_mode = 'graded'               # 'binary' is the EPO-faithful shape
    w_outcome = 1.0                      # terminal outcome added at the last turn
    done_tau = 0.6                       # dca threshold for "goal completed"
    reveal_threshold = 0.35              # FROZEN. Shared with every other baseline.
    leak_invalidates = True
    w_use, w_pool, w_close = 0.5, 0.3, 0.2
    w_halluc_pen = 0.5
    resolve_provenance = True

    # Records carry the action VECTORS only on request: 64 floats x 3 turns x 154
    # scenarios is a lot of repr() in a file people read by eye. The norms and the
    # turn-to-turn cosine are always stored, and they are what the diagnostics use.
    store_actions = False

    eval_split = 'test'
    seed = 1
