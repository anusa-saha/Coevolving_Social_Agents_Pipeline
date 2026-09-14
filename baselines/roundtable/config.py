"""Defaults for the round-table baseline.

Two backends, because the arm answers two different questions:

  local   Qwen2.5-7B-Instruct, the same model every other arm uses. This is the one that
          belongs in the comparison table -- a matched-capability floor, so a gap between
          it and PPDPP/EPO/Sotopia-RL is attributable to the method and not to the model.

  api     a frontier model through an OpenAI-compatible endpoint. Answers a different and
          also interesting question: how much of the benchmark is simply hard, versus how
          much is Qwen2.5-7B being a 7B. A win here is NOT evidence for any method, and it
          is not comparable with the other five arms. Run it alongside local, never
          instead.
"""
import os


class Defaults:
    # --- backend
    backend = 'local'                    # 'local' | 'api'
    model = 'Qwen/Qwen2.5-7B-Instruct'   # used when backend == 'local'
    api_model = os.environ.get('CSA_RT_API_MODEL', 'gpt-5.4-luna')
    api_base = os.environ.get('CSA_RT_API_BASE', '')   # '' = the client's default
    dtype = 'bfloat16'
    device = 'cuda:0'

    # --- conversation
    rounds = 2                           # passes around the table before settling
    max_new_tokens = 96
    settlement_tokens = 512              # a settlement JSON truncates at 96
    temperature = 0.7

    # --- how the table converges
    decide = 'chair'                     # 'chair' | 'vote' | 'converge'
    converge_rounds = 2                  # extra passes when decide == 'converge'

    # --- scoring. FROZEN and shared with every other arm; changing either makes this
    # arm's numbers incomparable with the rest of the repo.
    reveal_threshold = 0.35
    done_tau = 0.6

    eval_split = 'test'
    seed = 1


def apply_cli(cli):
    """Override the defaults from parsed CLI args, for the flags that exist."""
    for flag in ('backend', 'model', 'api_model', 'api_base', 'device', 'rounds',
                 'max_new_tokens', 'temperature', 'decide', 'converge_rounds', 'seed'):
        v = getattr(cli, flag, None)
        if v is not None:
            setattr(Defaults, flag, v)
    return Defaults
