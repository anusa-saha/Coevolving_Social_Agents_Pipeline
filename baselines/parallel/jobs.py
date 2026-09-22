"""Every pipeline stage of every arm, as jobs for run_all.py.

A job is one command in one arm's folder, with the GPUs it needs, the jobs that must have
succeeded first, a rough duration, and a priority that only breaks ties -- run_all.py
orders ready jobs by the longest chain of work still ahead of them. The durations are ESTIMATES -- scaled
from the README's figures by the larger split (363/33/154), assuming Qwen3.5-9B with the
fast linear-attention kernels on an A5000 -- and are used only to order work and to
backfill idle GPUs. run_all.py replaces them with measured durations as jobs finish.

Commands follow the README, with three decisions made up front so nothing needs a key or a
person mid-run:

  * EPO stage 1 calls the annotator API only if OPENROUTER_API_KEY is set when that job
    starts; otherwise it uses --fallback_only.
  * Sotopia-RL's GRPO reward source is picked by the README's gate: the reward model if
    its held-out pair-ranking accuracy is >= 0.60, lookahead otherwise.
  * SOTOPIA-Omega builds corpus B (local expert), and does not start if the probe printed
    STOP. run_all.py --force_gates overrides that.

Every training stage gets --grad_checkpointing: without it a LoRA backward on a 9B model
does not fit beside its own weights on 24 GB.
"""
import json
import os

MODEL = 'Qwen/Qwen3.5-9B'
RM_GATE = 0.60              # README: below this the reward model is at chance
TOM_STRATEGIES = ('stripped', 'basic', 'cot', 'tom_coach', 'tom_belief')


class Job(object):
    def __init__(self, name, cwd, cmd, gpus=1, deps=(), after=(), hours=1.0, priority=0,
                 gate=None, online=False, always=False, note=''):
        self.name = name
        self.arm = name.split('.', 1)[0]
        self.cwd = cwd
        self.cmd = cmd                  # list of args, or a callable returning one
        self.gpus = gpus                # 0 = CPU job
        self.deps = tuple(deps)         # must have SUCCEEDED
        self.after = tuple(after)       # must have FINISHED, whatever the outcome
        self.hours = hours
        self.priority = priority
        self.gate = gate                # callable(ctx) -> reason to hold, or None
        self.online = online            # talks to the Hugging Face Hub
        self.always = always            # re-run whenever any other selected job runs
        self.note = note

    def argv(self, python):
        cmd = self.cmd() if callable(self.cmd) else self.cmd
        return [python] + list(cmd)


def sr_reward_source(root):
    """(source, pair_rank): the README's gate on the trained reward model."""
    path = os.path.join(root, 'sotopia_rl', 'ckpt', 'rm', 'rm_meta.json')
    try:
        with open(path, encoding='utf-8') as f:
            rank = json.load(f).get('best_pair_rank')
    except (OSError, ValueError):
        rank = None
    if isinstance(rank, (int, float)) and rank >= RM_GATE:
        return 'rm', rank
    return 'lookahead', rank


def hf_cache():
    try:
        from huggingface_hub import constants
        return constants.HF_HUB_CACHE
    except Exception:                                # noqa: BLE001
        return os.path.join(os.path.expanduser('~'), '.cache', 'huggingface', 'hub')


def _omega_probe_gate(ctx):
    if 'STOP.' in ctx.log_text('om.probe'):
        return ('the probe printed STOP: slow mode did not improve disclosure, so the '
                'corpus would be plain self-play (README, SOTOPIA-Omega stage 0)')
    return None


def build(root):
    R = lambda *p: os.path.join(root, *p)            # noqa: E731
    cache = hf_cache()
    probe = 'setup.probe'
    jobs = []
    add = jobs.append

    # ------------------------------------------------------------------ setup
    add(Job('setup.prefetch', R('parallel'), ['preflight.py', '--prefetch'], gpus=0,
            hours=0.5, priority=1000, online=True,
            note='versions, kernels, data, model downloads'))
    add(Job('setup.probe', R('parallel'), ['preflight.py', '--probe'],
            deps=['setup.prefetch'], hours=0.2, priority=1000,
            note='peak memory of a LoRA step, generation speed, loss equivalence'))

    # ------------------------------------------------------------------ PPDPP
    P = R('ppdpp')
    add(Job('ppdpp.export', P, ['export_csa.py', '--out_dir', './data'], gpus=0,
            hours=0.02, priority=500))
    add(Job('ppdpp.sft_data', P, ['filter_sft_split.py'], gpus=0, hours=0.01,
            priority=500, note='archived planner SFT data, RL-train scenarios only'))
    add(Job('ppdpp.sft', P, [
        'sft.py', '--data_name', 'csa', '--model_name', 'roberta',
        '--model_name_or_path', 'roberta-large', '--data_dir', 'data_sft',
        '--output_dir', 'sft', '--do_train', '--do_eval', '--overwrite_output_dir',
        '--num_train_epochs', '10', '--max_seq_length', '512', '--gpu', '0',
        '--cache_dir', cache],
        deps=[probe, 'ppdpp.sft_data'], hours=0.5, priority=850))
    for reward, hours, prio in (('verifier', 34.0, 700), ('critic', 44.0, 690)):
        add(Job('ppdpp.rl_' + reward, P, [
            'run.py', '--data_name', 'csa', '--system', 'qwen', '--user', 'qwen',
            '--critic', 'qwen', '--csa_reward', reward, '--seed', '1', '--epochs', '6',
            '--max_turn', '8', '--do_train', '--do_eval', '--qwen_device_map', 'cuda:0',
            '--cache_dir', cache],
            deps=[probe, 'ppdpp.export', 'ppdpp.sft'], hours=hours, priority=prio,
            note='evaluates on the test split after every step'))

    # ------------------------------------------------------------------ EPO
    E = R('epo')

    def strategies(split):
        def cmd():
            extra = [] if os.environ.get('OPENROUTER_API_KEY') else ['--fallback_only']
            return ['make_strategies.py', '--split', split] + extra
        return cmd

    for split in ('train', 'valid'):
        add(Job('epo.strategies_' + split, E, strategies(split), gpus=0,
                deps=['ppdpp.sft_data'], hours=0.3, priority=500,
                note='API if OPENROUTER_API_KEY is set, --fallback_only otherwise'))
    add(Job('epo.sft', E, [
        'sft_epo.py', '--epochs', '3', '--lr', '1e-5', '--accum', '8',
        '--class_balance', 'sqrt_inverse', '--strategist_device', 'cuda:0',
        '--grad_checkpointing'],
        deps=[probe, 'epo.strategies_train', 'epo.strategies_valid'], hours=1.0,
        priority=870))
    add(Job('epo.rl', E, [
        'run_epo.py', '--episodes', '700', '--seed', '1', '--prm', 'verifier',
        '--prm_mode', 'binary', '--advantage', 'group', '--eval_every', '175',
        '--eval_split', 'test', '--agent_device', 'cuda:0', '--strategist_device',
        'cuda:1', '--grad_checkpointing'],
        gpus=2, deps=['epo.sft'], hours=22.0, priority=860,
        note='frozen agent on one card, trained strategist on the other'))

    # ------------------------------------------------------------------ Sotopia-RL
    S = R('sotopia_rl')
    add(Job('sr.collect_train', S, ['collect_episodes.py', '--split', 'train', '--k', '6',
                                    '--keep', '2'],
            deps=[probe], hours=27.0, priority=900))
    add(Job('sr.collect_valid', S, ['collect_episodes.py', '--split', 'valid', '--k', '6',
                                    '--keep', '2'],
            deps=[probe], hours=2.5, priority=600))
    add(Job('sr.rm_data', S, ['make_rm_data.py', '--episodes', 'data/episodes-train.jsonl'],
            gpus=0, deps=['sr.collect_train'], hours=0.05, priority=580))
    add(Job('sr.sft', S, ['train_sft.py', '--epochs', '3', '--lr', '1e-4', '--accum', '8',
                          '--grad_checkpointing'],
            deps=['sr.collect_train', 'sr.collect_valid'], hours=3.0, priority=580))
    add(Job('sr.rm', S, ['train_rm.py', '--epochs', '8', '--lr', '5e-6', '--holdout',
                         '0.15', '--grad_checkpointing'],
            deps=['sr.rm_data'], hours=6.0, priority=570))

    def grpo():
        src, _rank = sr_reward_source(root)
        cmd = ['train_grpo.py', '--adapter', 'ckpt/sft', '--reward_source', src,
               '--groups', '175', '--group', '8', '--kl_beta', '0.02', '--seed', '1',
               '--grad_checkpointing']
        return cmd + (['--rm', 'ckpt/rm'] if src == 'rm' else [])

    add(Job('sr.grpo', S, grpo, deps=['sr.sft', 'sr.rm'], hours=17.0, priority=560,
            note='reward source chosen by the RM gate when the job starts'))
    add(Job('sr.eval_base', S, ['evaluate_sr.py', '--adapter', '', '--split', 'test',
                                '--tag', 'base'],
            deps=[probe], hours=2.0, priority=250))
    add(Job('sr.eval_sft', S, ['evaluate_sr.py', '--adapter', 'ckpt/sft', '--split', 'test',
                               '--tag', 'sft'],
            deps=['sr.sft'], hours=2.0, priority=400))
    add(Job('sr.eval_grpo', S, lambda: [
        'evaluate_sr.py', '--adapter',
        'ckpt/grpo/grpo-%s-seed1/final' % sr_reward_source(root)[0],
        '--split', 'test', '--tag', 'grpo'],
        deps=['sr.grpo'], hours=2.0, priority=400))

    # ------------------------------------------------------------------ Sotopia-ToM
    T = R('sotopia_tom')
    for i, s in enumerate(TOM_STRATEGIES):
        add(Job('tom.' + s, T, ['run_tom.py', '--strategies', s, '--split', 'test',
                                '--out', os.path.join('logs', 'summary-%s.json' % s)],
                deps=[probe], hours=2.5 if s.startswith('tom_') else 2.0,
                priority=300 - i))
    add(Job('tom.compare', T, ['run_tom.py', '--compare', '--split', 'test'], gpus=0,
            deps=['tom.' + s for s in TOM_STRATEGIES], hours=0.02, priority=300))

    # ------------------------------------------------------------------ SOTOPIA-Omega
    O = R('sotopia_omega')
    add(Job('om.probe', O, ['generate_omega.py', '--split', 'train', '--probe', '5',
                            '--expert', 'local'],
            deps=[probe], hours=0.3, priority=890))
    add(Job('om.corpus_train', O, [
        'generate_omega.py', '--split', 'train', '--expert', 'local', '--k', '6',
        '--keep', '2', '--stall_after', '1', '--stall_patience', '1', '--seed', '1'],
        deps=['om.probe'], hours=36.0, priority=880, gate=_omega_probe_gate))
    add(Job('om.corpus_valid', O, ['generate_omega.py', '--split', 'valid', '--expert',
                                   'local', '--seed', '1'],
            deps=['om.probe'], hours=3.5, priority=590, gate=_omega_probe_gate))
    add(Job('om.sft', O, ['train_sft_om.py', '--mode_filter', 'all', '--grad_checkpointing'],
            deps=['om.corpus_train', 'om.corpus_valid'], hours=3.0, priority=550))
    add(Job('om.eval_base', O, ['evaluate_om.py', '--adapter', '', '--split', 'test',
                                '--tag', 'base'],
            deps=[probe], hours=2.0, priority=250))
    for tag, extra, hours in (('omega', [], 2.0),
                              ('omega-adaptive', ['--eval_mode', 'adaptive'], 3.0),
                              ('omega-withhold', ['--opponent', 'withhold'], 2.0)):
        add(Job('om.eval_' + tag.replace('omega-', '').replace('omega', 'sft'), O,
                ['evaluate_om.py', '--adapter', 'ckpt/sft', '--split', 'test',
                 '--tag', tag] + extra,
                deps=['om.sft'], hours=hours, priority=400))

    # ------------------------------------------------------------------ Round Table
    add(Job('rt.chair', R('roundtable'), ['run_rt.py', '--backend', 'local', '--decide',
                                          'chair', '--split', 'test'],
            deps=[probe], hours=1.0, priority=240))

    # ------------------------------------------------------------------ analysis
    produces = [j.name for j in jobs
                if j.name.startswith(('ppdpp.rl_', 'sr.eval_', 'om.eval_'))
                or j.name in ('epo.rl', 'tom.compare', 'rt.chair')]
    add(Job('analysis.metrics', R('analysis'), ['compute_extended_metrics.py'], gpus=0,
            after=produces, hours=0.05, priority=100, always=True,
            note='after every arm finishes, success or not; re-runs when anything ran'))
    return jobs
