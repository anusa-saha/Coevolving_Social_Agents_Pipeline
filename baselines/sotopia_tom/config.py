"""Defaults. One frozen model, no training anywhere in this arm."""


class Defaults:
    # Qwen2.5-7B-Instruct throughout, matching the other three baselines exactly. It
    # plays the chair AND the advisors, so the environment half of the comparison is
    # identical and any difference is attributable to the prompting strategy.
    model = 'Qwen/Qwen2.5-7B-Instruct'
    dtype = 'bfloat16'
    device = 'cuda:0'

    max_new_tokens = 96
    settlement_max_tokens = 512      # a settlement JSON truncates at 96
    cot_extra_tokens = 160           # the CoT arm emits THINKING before TURN
    tom_max_tokens = 220             # analyst table / belief-state JSON

    # Evaluation is greedy so the numbers are reproducible; temperature is only used if
    # someone deliberately turns sampling on.
    temperature = 0.7

    reveal_threshold = 0.35          # FROZEN. Shared with every other baseline.
    done_tau = 0.6                   # dca threshold for "goal completed"
    eval_split = 'test'
    seed = 1
