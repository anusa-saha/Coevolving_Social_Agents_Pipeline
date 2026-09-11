"""Defaults for the Omega port."""


class Defaults:
    # --- expert / generator
    expert = 'local'                 # 'local' | 'api'
    expert_model = 'Qwen/Qwen2.5-7B-Instruct'
    api_base = ''                    # e.g. an OpenAI-compatible gateway; '' = default
    dtype = 'bfloat16'
    device = 'cuda:0'

    max_new_tokens = 96
    settlement_tokens = 512          # a settlement JSON truncates at 96
    stage_tokens = 160               # one slow-mode reasoning stage
    temperature = 0.9                # >0 or the k rollouts of a scenario are identical

    # --- stall detection. Computable, no judge, nothing to tune against a threshold.
    # "step >= stall_after AND decisive pooling has not advanced for stall_patience
    # chair turns". With max_turn typically 3-4, these values mean the switch can fire
    # from the second chair turn onward and will fire by the third if nothing lands.
    stall_after = 1
    stall_patience = 1

    # --- corpus
    rollout_k = 6                    # rollouts per scenario
    rollout_keep = 2                 # top-N kept, ranked WITHIN the scenario so hard
                                     # scenarios still contribute
    opponent = 'none'                # 'none' | 'withhold'

    # --- student (SFT). Same model family as every other arm, so the comparison holds.
    student_model = 'Qwen/Qwen2.5-7B-Instruct'
    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.05
    lora_targets = ('q_proj', 'k_proj', 'v_proj', 'o_proj',
                    'gate_proj', 'up_proj', 'down_proj')
    max_len = 1536
    sft_lr = 1e-4
    sft_epochs = 3
    sft_accum = 8

    # --- scoring
    reveal_threshold = 0.35          # FROZEN. Shared with every other baseline.
    done_tau = 0.6
    eval_split = 'test'
    seed = 1
