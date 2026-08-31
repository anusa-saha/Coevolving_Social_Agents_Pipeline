"""Every default in one place, with the reason where it departs from the paper.

Sotopia-RL's published settings assume ~2,050 SOTOPIA training episodes and GPT-4o
demonstrations. This runs on 99 CSA scenarios with Qwen self-play, so several numbers are
deliberately different. See sotopia-rl-vs-vanilla.pdf for the full argument.
"""


class Defaults:
    # --- the meeting agents. Frozen throughout: only the chair policy is trained, and
    # advisors trained on the same reward collapse the hidden profile.
    agent_model = 'Qwen/Qwen2.5-7B-Instruct'
    agent_dtype = 'bfloat16'
    agent_device = 'cuda:0'
    agent_max_new_tokens = 96
    settlement_max_tokens = 512          # a settlement JSON truncates at 96
    agent_temperature = 0.7

    # --- the trained chair policy and the reward model
    # Both are the SAME checkpoint as the agent, so exactly one copy is resident and the
    # three roles are LoRA adapters over it (models_sr.SharedBackbone). Three separate
    # bf16 7B models would be ~45 GB of weights; this is ~15 GB plus ~300 MB of adapters.
    # There is therefore ONE device setting -- agent_device -- and no policy/rm device.
    lora_r = 16                          # paper full-FTs; 99 scenarios would memorise
    lora_alpha = 32
    lora_dropout = 0.05
    lora_targets = ('q_proj', 'k_proj', 'v_proj', 'o_proj',
                    'gate_proj', 'up_proj', 'down_proj')
    max_len = 1536

    # --- stage 1: episode collection
    rollout_k = 6                        # rollouts per scenario
    rollout_keep = 2                     # top-N kept, ranked WITHIN the scenario so hard
                                         # scenarios still contribute
    rollout_temperature = 0.9            # must be > 0 or the k rollouts are identical

    # --- stage 2.1: behaviour cloning
    sft_lr = 1e-4                        # paper's value; LoRA tolerates it
    sft_epochs = 3                       # paper runs 500; 99 scenarios do not survive that
    sft_accum = 8

    # --- stage 2.2: reward model (MSE regression on attributed scalars)
    rm_lr = 1e-5
    rm_epochs = 4                        # paper runs 30; ~1.2k turns would memorise
    rm_accum = 8

    # --- stage 2.3: GRPO
    grpo_lr = 5e-6
    grpo_group = 8                       # paper uses 16; each candidate costs a generation
    grpo_groups = 175                    # 175 x 8 = 1400 chair generations
    grpo_kl_beta = 0.02                  # paper reports a KL penalty; keep it small
    grpo_clip = 1.0
    grpo_groups_per_update = 1
    # 'lookahead' needs NO reward model at all -- it commits a candidate, lets one advisor
    # answer, and reads the disclosure detector. On a tight card that removes the rm
    # adapter and the whole train_rm stage; it is also deterministic and API-free.
    reward_source = 'rm'                 # 'rm' | 'lookahead' | 'hybrid'
    grad_checkpointing = False           # ~6x less activation memory, ~30% slower

    # --- reward design
    dims = ('pool', 'use', 'cover')
    dim_weights = None                   # None -> equal, as the paper uses
    within_episode_norm = False          # see attribution.Normaliser docstring
    gate_on_integrity = True             # leak or invalid schema -> all-zero labels
    reveal_threshold = 0.35              # FROZEN. Shared with every other baseline.

    # --- evaluation
    eval_split = 'test'
    done_tau = 0.6                       # dca threshold for "goal completed"
    seed = 1


def override(cfg, args, mapping):
    """Copy argparse values onto the config object where the flag was given."""
    for flag, attr in mapping.items():
        v = getattr(args, flag, None)
        if v is not None:
            setattr(cfg, attr, v)
    return cfg
