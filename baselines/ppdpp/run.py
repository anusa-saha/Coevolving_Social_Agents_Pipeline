import sys
import os
import shutil
from env import Env, LOCAL_BACKENDS
from agent import PPDPP
from utils import *
from itertools import count
from tqdm import tqdm
import argparse
from transformers import BertTokenizer, RobertaTokenizer, BertConfig, RobertaConfig
_CSA_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CSA_ROOT not in sys.path:
    sys.path.insert(0, _CSA_ROOT)
from csa_core.verifier import floor_score
from csa_core import runlog

# Conversation and rollout logs (csa only), beside the scripts like every other arm's.
LOGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
try:
    from fastchat.model import add_model_args
except ImportError:
    # fastchat serves only the vicuna and llama2 backends. Register the same arguments
    # it would have, so the rest of the parser is unchanged.
    def add_model_args(parser):
        parser.add_argument('--device', type=str,
                            default='cuda' if torch.cuda.is_available() else 'cpu',
                            choices=['cpu', 'cuda', 'mps', 'xpu', 'npu'])
        parser.add_argument('--max-gpu-memory', dest='max_gpu_memory', default=None)
        parser.add_argument('--load-8bit', dest='load_8bit', action='store_true')
        parser.add_argument('--cpu-offloading', dest='cpu_offloading', action='store_true')

tok = {'bert': BertTokenizer, 'roberta': RobertaTokenizer}
cfg = {'bert': BertConfig, 'roberta': RobertaConfig}

# Windows defaults stdout to cp1252 when redirected to a file. The environment prints
# every generated utterance, so one character outside that codepage -- an en dash, a
# curly quote -- raises UnicodeEncodeError and kills a multi-hour run mid-episode.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass



def _csa_record(env, epi_reward, done, turns):
    """Everything the metric catalogue needs, so any number can be recomputed offline
    without re-running a model. One definition for evaluation records and training
    rollouts, so the two logs cannot disagree about what an episode was."""
    return {
        'dialog': list(env.conversation), 'reward': epi_reward,
        'uid': env.case.get('uid'),
        'domain': env.case.get('domain'),
        'num_agents': env.case.get('num_agents'),
        'scenario_type': env.case.get('scenario_type'),
        'settlement': env.settlement,
        'settle_turn': env.settle_turn,
        'settled_by': env.settled_by,
        'score': env.last_score,
        'score_norm': env.last_score_norm,
        'floor': floor_score(env.case),
        'revealed': sorted(env.revealed),
        'reveal_elicited': dict(env.reveal_elicited),
        'addressed': sorted(env.addressed),
        'leaks': list(env.leaks),
        'done': done,
        # --- efficiency and cost (catalogue section C) ---
        'turns': turns,
        'max_turn': env.max_turn,
        'n_calls': getattr(env, 'n_calls', 0),
        'calls_by_role': dict(getattr(env, 'calls_by_role', {})),
        'prompt_chars': getattr(env, 'prompt_chars', 0),
        # --- policy diagnostics (catalogue section G) ---
        'act_history': list(getattr(env, 'act_history', [])),
        'reveal_turn': dict(getattr(env, 'reveal_turn', {})),
    }


def _csa_record_ep(ep, epi_reward, done, turns):
    """Same shape as _csa_record, reading a batched-rollout episode object (env.py's
    _new_csa_episode) instead of the single-episode Env instance. Kept as a separate
    function rather than a branch inside _csa_record so neither path has to guess which
    kind of object it was handed."""
    return {
        'dialog': list(ep.conversation), 'reward': epi_reward,
        'uid': ep.case.get('uid'),
        'domain': ep.case.get('domain'),
        'num_agents': ep.case.get('num_agents'),
        'scenario_type': ep.case.get('scenario_type'),
        'settlement': ep.settlement,
        'settle_turn': ep.settle_turn,
        'settled_by': ep.settled_by,
        'score': ep.last_score,
        'score_norm': ep.last_score_norm,
        'floor': floor_score(ep.case),
        'revealed': sorted(ep.revealed),
        'reveal_elicited': dict(ep.reveal_elicited),
        'addressed': sorted(ep.addressed),
        'leaks': list(ep.leaks),
        'done': done,
        'turns': turns,
        'max_turn': ep.max_turn,
        'n_calls': getattr(ep, 'n_calls', 0),
        'calls_by_role': dict(getattr(ep, 'calls_by_role', {})),
        'prompt_chars': getattr(ep, 'prompt_chars', 0),
        'act_history': list(getattr(ep, 'act_history', [])),
        'reveal_turn': dict(getattr(ep, 'reveal_turn', {})),
    }


def _run_batched_rollout(args, env, policy, rollouts, train_step, i_episode_start):
    """One args.max_steps iteration's worth of sampling (args.sample_times episodes),
    run in parallel batches of args.rollout_batch instead of one episode at a time.

    Only the environment simulation and the policy's ROLLOUT forward pass (action
    selection, no_grad) are batched -- see PPDPP.select_action_batch's docstring.
    Learning happens afterward, per episode: each episode's trajectory of
    (state, action_idx) pairs is replayed through PPDPP.logprob_of_action -- a fresh,
    single-episode forward pass -- immediately before that episode's own independent
    backward()/optimizer.step(). That replay is what makes each episode's REINFORCE
    update genuinely independent, with the same math as the sequential loop in
    train(): a batched forward call shares one computation graph across every episode
    active in that tick, and splitting that graph's log_probs apart for separate
    per-episode backward() calls corrupts on the second one -- PyTorch frees a
    graph's buffers after its first backward(), so whichever episode shared a tick
    with the one that just backpropped hits already-freed buffers.
    """
    SR, AvgT, total_reward = 0., 0., 0.
    loss = torch.tensor(0, dtype=torch.float, device=args.device)
    remaining = args.sample_times
    i_episode = i_episode_start
    while remaining > 0:
        b = min(args.rollout_batch, remaining)
        remaining -= b
        print('\n================new batch of {} tuples===================='.format(b))
        states = env.reset_batch(b)
        trajectories = [[] for _ in range(b)]  # per episode: [(state, action_idx), ...]
        rewards = [[] for _ in range(b)]
        epi_reward = [0.0] * b
        done_flags = [0] * b
        turns = [0] * b
        active = list(range(b))
        t = 0
        while active:
            sub_states = [states[i] for i in active]
            actions, actions_idx = policy.select_action_batch(sub_states, is_test=False)
            action_for_env = [None] * b
            idx_for_i = {}
            state_for_i = {}
            for k, i in enumerate(active):
                action_for_env[i] = actions[k]
                idx_for_i[i] = actions_idx[k]
                state_for_i[i] = sub_states[k]

            results = env.step_batch(action_for_env)
            just_finished = []
            for i in active:
                state_i, reward_i, done_i = results[i]
                # The state BEFORE this turn's action, not after -- what select_action
                # was actually conditioned on -- so logprob_of_action's replay scores
                # the same (state, action) pair the rollout sampled.
                trajectories[i].append((state_for_i[i], idx_for_i[i]))
                states[i] = state_i
                rewards[i].append(reward_i)
                epi_reward[i] += reward_i
                turns[i] = t + 1
                if done_i:
                    done_flags[i] = done_i
                    just_finished.append(i)
            for i in just_finished:
                active.remove(i)
            t += 1

        for i in range(b):
            log_probs_i = [policy.logprob_of_action(s, a) for s, a in trajectories[i]]
            newloss = policy.optimize_from_buffer(log_probs_i, rewards[i])
            if newloss is not None:
                loss = loss + newloss
            if done_flags[i] == 1:
                SR += 1
            AvgT += turns[i]
            total_reward += epi_reward[i]
            if rollouts is not None:
                ep = env.batch[i]
                rollouts.add(ep.case, _csa_record_ep(ep, epi_reward[i], done_flags[i], turns[i]),
                             step=train_step, candidate=i_episode, reward=epi_reward[i],
                             step_rewards=list(rewards[i]),
                             loss=None if newloss is None else float(newloss))
            i_episode += 1
        # Each batch's KV cache and padded activations are freed once the batch's
        # episodes are all done, but the allocator caches those blocks rather than
        # returning them to the driver. Freed-but-cached blocks are still fine to reuse
        # within one run, but a training step that follows can want a different padded
        # shape than what's cached and end up fragmenting instead of reusing it --
        # emptying the cache here trades a small amount of time for headroom on a card
        # this close to the model's footprint.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return SR, AvgT, total_reward, loss


def _run_batched_eval(args, test_env, policy, elog, rec_file, i_episode):
    """Batched sibling of evaluate()'s per-episode loop -- same eval semantics (greedy
    policy, is_test=True, every test case scored exactly once, same record/elog output)
    but scoring `eval_batch` test cases per padded Qwen generate() call instead of one
    case at a time. No optimizer step anywhere here; this only speeds up the rollout,
    identically to _run_batched_rollout on the training side.

    Returns (SR, AvgT, total_reward, SR_turn) with the same meaning as the caller's
    locals in the sequential evaluate() loop, so evaluate() can finish exactly as before
    (mean/turn-success computation, save_rl_mtric, printing) whichever path produced them.
    """
    cases = list(test_env.dataset)
    test_size = len(cases)
    eval_batch = max(1, getattr(args, 'eval_batch', 0) or args.rollout_batch)
    SR, AvgT, total_reward = 0, 0, 0
    SR_turn = [0] * args.max_turn
    for start in tqdm(range(0, test_size, eval_batch), desc='batched eval'):
        chunk = cases[start:start + eval_batch]
        b = len(chunk)
        states = test_env.reset_batch_fixed(chunk)
        rewards = [[] for _ in range(b)]
        epi_reward = [0.0] * b
        done_flags = [0] * b
        turns = [0] * b
        active = list(range(b))
        t = 0
        while active:
            sub_states = [states[i] for i in active]
            actions = policy.select_action_batch(sub_states, is_test=True)
            action_for_env = [None] * b
            for k, i in enumerate(active):
                action_for_env[i] = actions[k]

            results = test_env.step_batch(action_for_env)
            just_finished = []
            for i in active:
                state_i, reward_i, done_i = results[i]
                states[i] = state_i
                rewards[i].append(reward_i)
                epi_reward[i] += reward_i
                turns[i] = t + 1
                if done_i:
                    done_flags[i] = done_i
                    just_finished.append(i)
            for i in just_finished:
                active.remove(i)
            t += 1

        for i in range(b):
            if done_flags[i] == 1:
                SR_turn = [v + 1 if k > turns[i] - 1 else v for k, v in enumerate(SR_turn)]
                SR += 1
            AvgT += turns[i]
            total_reward += epi_reward[i]
            ep = test_env.batch[i]
            record = {'dialog': list(ep.conversation), 'reward': epi_reward[i]}
            record.update(_csa_record_ep(ep, epi_reward[i], done_flags[i], turns[i]))
            elog.add(ep.case, record)
            rec_file.write('%s\n\n' % str(record))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return SR, AvgT, total_reward, SR_turn


def _latest_checkpoint_epoch(args, filename):
    """Highest RL-agent checkpoint epoch already saved for this run's filename, or 0
    if none. Checkpoints are named '<filename>-epoch-<N>' (see PPDPP.save_model and
    the pruning logic in train()) -- deterministic given (data_name, sft_dir, system,
    user, critic, csa_reward, seed), so a rerun with the same args finds the same
    checkpoints a previous, interrupted run of this exact config left behind."""
    ck_root = os.path.join(TMP_DIR[args.data_name], 'RL-agent')
    if not os.path.isdir(ck_root):
        return 0
    prefix = filename + '-epoch-'
    best = 0
    for d in os.listdir(ck_root):
        if not (d.startswith(prefix) and os.path.isdir(os.path.join(ck_root, d))):
            continue
        try:
            best = max(best, int(d[len(prefix):]))
        except ValueError:
            continue
    return best


def _eval_output_exists(args, filename, i_episode):
    """Whether evaluate()'s output file for this i_episode is already on disk -- same
    filename it writes at the end of evaluate() (both the batched and sequential
    paths write here identically)."""
    test_filename = 'Evaluate-epoch-{}-'.format(i_episode) + filename
    path = os.path.join(TMP_DIR[args.data_name], 'eval_result', test_filename + '.txt')
    return os.path.isfile(path)


def train(args, config, dataset, filename, tokenizer):
    env = Env(args, dataset, mode='train') # env init
    set_random_seed(args.seed)
    policy = PPDPP(args, config, tokenizer) # policy network init

    # load policy parameters
    if args.sft_dir is not None:
        print('Staring loading policy model from {}'.format(args.sft_dir))
        policy.load_model(data_name=args.data_name, filename=args.sft_dir)

    # --resume: same (data_name, sft_dir, system, user, critic, csa_reward, seed) as a
    # previous, interrupted run of this script produces the same `filename`, so its
    # checkpoints and eval output are found here rather than redone. Only kicks in
    # when --load_rl_epoch wasn't already given explicitly, so an explicit request to
    # load a *specific* epoch is never second-guessed.
    if args.resume and args.load_rl_epoch == 0:
        detected = _latest_checkpoint_epoch(args, filename)
        if detected > 0:
            print('[resume] found a checkpoint at epoch {} for this config; resuming '
                  'from there instead of epoch 0'.format(detected))
            args.load_rl_epoch = detected

    if args.load_rl_epoch > 0:
        print('Staring loading rl model in epoch {}'.format(args.load_rl_epoch))
        policy.load_model(data_name=args.data_name, filename=filename, epoch_user=args.load_rl_epoch)

    test_performance = []
    if args.do_eval:
        if args.resume and _eval_output_exists(args, filename, 0):
            print('[resume] epoch-0 evaluation output already exists for this config; '
                  'skipping the pre-training eval pass rather than redoing it.')
        else:
            SR15_mean = evaluate(args, dataset, policy, filename, 0, env)
            test_performance = [SR15_mean]
    if not args.do_train:
        return
    rollouts = None
    if args.data_name == 'csa':
        rollouts = runlog.RolloutLog(LOGS, filename,
                                     algo='ppdpp-reinforce-%s' % args.csa_reward)

    # Batched rollout needs a single shared model doing every role's generation, so
    # unrelated episodes' turns can share one padded generate() call. Both reward arms
    # are batched in step_batch -- the critic arm's per-episode LLM-judge draw is
    # batched across episodes too (_qwen_critic_reward_batch). Outside data_name/model
    # combination this falls back to the untouched sequential loop below, so nothing
    # else changes behavior.
    use_batched_rollout = (
        args.data_name == 'csa' and args.rollout_batch > 1 and
        args.system == 'qwen' and args.user == 'qwen' and args.critic == 'qwen'
    )

    # Whenever a checkpoint's weights were loaded -- whether --resume auto-detected it
    # above or --load_rl_epoch named it explicitly -- continue step numbering from
    # there instead of restarting at 1. Restarting at 1 after loading epoch N's weights
    # would re-run and overwrite epoch 1..N's checkpoint/eval files with a run that
    # actually started from epoch N, silently discarding what they recorded.
    # --max_steps stays the absolute target step count either way, not "how many more
    # steps to run".
    start_step = args.load_rl_epoch + 1 if args.load_rl_epoch > 0 else 1
    if start_step > 1:
        print('[resume] training steps 1..{} already done; continuing from step {}'
              .format(start_step - 1, start_step))
    for train_step in range(start_step, args.max_steps+1):
        if use_batched_rollout:
            SR, AvgT, total_reward, loss = _run_batched_rollout(
                args, env, policy, rollouts, train_step, i_episode_start=0)
            enablePrint()
            print('loss : {} in epoch_uesr {}'.format(loss.item()/args.sample_times, args.sample_times))
            print('SR:{}, AvgT:{}, rewards:{} Total epoch_uesr:{}'.format(SR / args.sample_times,
                        AvgT / args.sample_times, total_reward / args.sample_times, args.sample_times))
            frac = policy.degenerate_fraction()
            print('zero-gradient updates: {:.1%} ({}/{})'.format(
                frac, getattr(policy, 'degenerate_updates', 0),
                getattr(policy, 'total_updates', 0)))
            if frac > 0.5:
                print('  ^ over half of all updates had zero advantage. This run is not '
                      'training. Fix the reward before reading anything into the curve.')
            if train_step % args.eval_num == 0:
                SR_all = evaluate(args, dataset, policy, filename, train_step, env)
                test_performance.append(SR_all)
            if train_step % args.save_num == 0:
                policy.save_model(data_name=args.data_name, filename=filename, epoch_user=train_step)
                try:
                    ck_root = os.path.join(TMP_DIR[args.data_name], 'RL-agent')
                    dirs = [os.path.join(ck_root, d) for d in os.listdir(ck_root)
                            if d.startswith(filename + '-epoch-')]
                    dirs = sorted([d for d in dirs if os.path.isdir(d)],
                                  key=os.path.getmtime, reverse=True)
                    for old in dirs[max(1, args.keep_ckpt):]:
                        shutil.rmtree(old, ignore_errors=True)
                        print('[janitor] removed old checkpoint %s' % os.path.basename(old))
                except Exception as e:
                    print('[janitor] skipped: %s' % e)
            continue

        SR, AvgT, total_reward = 0., 0., 0.
        loss = torch.tensor(0, dtype=torch.float, device=args.device)
        for i_episode in tqdm(range(args.sample_times),desc='sampling'):
            #blockPrint()
            print('\n================new tuple:{}===================='.format(i_episode))
            state = env.reset()

            epi_reward = 0
            step_rewards = []
            done = False
            for t in count():   # user  dialog
                action = policy.select_action(state)
                state, reward, done = env.step(action)
                epi_reward += reward
                step_rewards.append(reward)
                reward = torch.tensor([reward], device=args.device, dtype=torch.float)
                policy.rewards.append(reward)

                if done:
                    if done == 1:
                        SR += 1
                    AvgT += t+1
                    total_reward += epi_reward
                    break

            newloss = policy.optimize_model()
            if newloss is not None:
                loss += newloss
            if rollouts is not None:
                rollouts.add(env.case, _csa_record(env, epi_reward, done, t + 1),
                             step=train_step, candidate=i_episode, reward=epi_reward,
                             step_rewards=step_rewards,
                             loss=None if newloss is None else float(newloss))
            
        enablePrint() # Enable print function
        print('loss : {} in epoch_uesr {}'.format(loss.item()/args.sample_times, args.sample_times))
        print('SR:{}, AvgT:{}, rewards:{} Total epoch_uesr:{}'.format(SR / args.sample_times,
                    AvgT / args.sample_times, total_reward / args.sample_times, args.sample_times))
        # A flat learning curve means nothing until you know whether the updates carried
        # any gradient at all. Print it beside the curve, every step.
        frac = policy.degenerate_fraction()
        print('zero-gradient updates: {:.1%} ({}/{})'.format(
            frac, getattr(policy, 'degenerate_updates', 0),
            getattr(policy, 'total_updates', 0)))
        if frac > 0.5:
            print('  ^ over half of all updates had zero advantage. This run is not '
                  'training. Fix the reward before reading anything into the curve.')

        if train_step % args.eval_num == 0:
            SR_all = evaluate(args, dataset, policy, filename, train_step, env)
            test_performance.append(SR_all)
        if train_step % args.save_num == 0:
            policy.save_model(data_name=args.data_name, filename=filename, epoch_user=train_step)
            # Retention: each RL checkpoint is ~1.05 GB and this disk has a few GB spare.
            # Without pruning, a 10-step run fills it and torch.save dies mid-write,
            # taking the whole run with it. Keep the newest `keep_ckpt` only.
            try:
                ck_root = os.path.join(TMP_DIR[args.data_name], 'RL-agent')
                # This run's checkpoints only. Parallel runs share RL-agent/, and pruning
                # across runs deletes the other run's checkpoints.
                dirs = [os.path.join(ck_root, d) for d in os.listdir(ck_root)
                        if d.startswith(filename + '-epoch-')]
                dirs = sorted([d for d in dirs if os.path.isdir(d)],
                              key=os.path.getmtime, reverse=True)
                for old in dirs[max(1, args.keep_ckpt):]:
                    shutil.rmtree(old, ignore_errors=True)
                    print('[janitor] removed old checkpoint %s' % os.path.basename(old))
            except Exception as e:
                print('[janitor] skipped: %s' % e)
    if rollouts is not None:
        rollouts.close()
    print(test_performance)

def evaluate(args, dataset, policy, filename, i_episode, train_env):
    # Any locally served backend must hand its weights to the test env: Env only loads a
    # model when mode == 'train', so a test env built without them gets None and dies on
    # the first generation. Upstream hardcoded vicuna/llama2 here; qwen has to be in the
    # same branch or it silently falls through.
    if set([args.system, args.user, args.critic]) & set(LOCAL_BACKENDS):
        test_env = Env(args, dataset, mode='test', env_model=train_env.vicuna_model, env_tokenizer=train_env.vicuna_tokenizer)
    else:
        test_env = Env(args, dataset, mode='test') # env init
    set_random_seed(args.seed)

    SR, AvgT, total_reward = 0, 0, 0
    SR_turn = [0]* args.max_turn
    turn_result = []
    result = []
    test_size = len(test_env.dataset)
    print('Test size: ', test_size)
    test_filename = 'Evaluate-epoch-{}-'.format(i_episode) + filename
    record_filename = 'Record-epoch-{}-'.format(i_episode) + filename
    REC_PATH = TMP_DIR[args.data_name] + '/eval_result/' + record_filename + '.txt'
    if not os.path.isdir(TMP_DIR[args.data_name] + '/eval_result/'):
        os.makedirs(TMP_DIR[args.data_name] + '/eval_result/')
    # Explicit utf-8: the default locale encoding writes unrepresentable characters as
    # mojibake, which then makes the record file undecodable when scoring it.
    rec_file = open(REC_PATH, 'w', encoding='utf-8')
    elog = runlog.EvalLog(LOGS, 'eval-' + record_filename) if args.data_name == 'csa' else None

    # Same precondition as the batched training rollout: one shared Qwen model doing
    # every role's generation. Both reward arms are batched (see step_batch).
    use_batched_eval = (
        args.data_name == 'csa' and args.rollout_batch > 1 and
        args.system == 'qwen' and args.user == 'qwen' and args.critic == 'qwen'
    )
    if use_batched_eval:
        SR, AvgT, total_reward, SR_turn = _run_batched_eval(
            args, test_env, policy, elog, rec_file, i_episode)
        enablePrint()
        if elog is not None:
            elog.close()
        SR_mean = float(SR) / test_size
        AvgT_mean = float(AvgT) / test_size
        reward_mean = total_reward / test_size
        SR_all = [SR_mean, AvgT_mean, reward_mean]
        save_rl_mtric(dataset=args.data_name, filename=test_filename, epoch=test_size - 1,
                      SR=SR_all, mode='test')
        print('save test evaluate successfully!')
        SRturn_all = [float(v) / test_size for v in SR_turn]
        print('success turn:{}'.format(SRturn_all))
        print('SR:{}, AvgT:{}, reward:{}'.format(SR_mean, AvgT_mean, reward_mean))
        PATH = TMP_DIR[args.data_name] + '/eval_result/' + test_filename + '.txt'
        with open(PATH, 'a') as f:
            f.write('Training epocch:{}\n'.format(i_episode))
            f.write('================================\n')
        with open(PATH, 'a') as f:
            f.write('{}\t{}\t{}\t{}\n'.format(i_episode, SR_mean, AvgT_mean, reward_mean))
        rec_file.close()
        return SR_all

    for test_num in tqdm(range(test_size)):  #test_size
        #blockPrint()
        print('\n================test tuple:{}===================='.format(test_num))
        epi_reward = 0
        done = 0
        is_last_turn = False
        state = test_env.reset()
        for t in count():  # user  dialog
            action = policy.select_action(state, is_test=True)
            state, reward, done = test_env.step(action)
            if args.data_name == 'cb' and reward < 0: # reward = Sale-to-List Ratio
                reward = 0
            epi_reward += reward

            if done:
                if done == 1:  
                    SR_turn = [v+1 if i>t  else v for i, v in enumerate(SR_turn) ]
                    SR += 1
                total_reward += epi_reward
                AvgT += t+1

                record = {'dialog':state, 'reward':epi_reward}
                if args.data_name == 'csa':
                    record.update(_csa_record(test_env, epi_reward, done, t + 1))
                    elog.add(test_env.case, record)
                rec_file.write('%s\n\n' % str(record))
                break

        enablePrint()
            
    
    if elog is not None:
        elog.close()
    SR_mean = float(SR)/test_size
    AvgT_mean = float(AvgT)/test_size
    reward_mean = total_reward/test_size
    SR_all = [SR_mean, AvgT_mean, reward_mean]
    save_rl_mtric(dataset=args.data_name, filename=test_filename, epoch=test_num, SR=SR_all, mode='test')  # save RL SR
    print('save test evaluate successfully!')

    SRturn_all = [0] * args.max_turn
    for i in range(len(SRturn_all)):
        SRturn_all[i] = float(SR_turn[i])/test_size
    print('success turn:{}'.format(SRturn_all))
    print('SR:{}, AvgT:{}, reward:{}'.format(SR_mean, AvgT_mean, reward_mean))
    PATH = TMP_DIR[args.data_name] + '/eval_result/' + test_filename + '.txt'
    with open(PATH, 'a') as f:
        f.write('Training epocch:{}\n'.format(i_episode))
        f.write('===========Test Turn===============\n')
        f.write('Testing {} user tuples\n'.format(test_num))
        for i in range(len(SRturn_all)):
            f.write('Testing SR-turn@{}: {}\n'.format(i, SRturn_all[i]))
        f.write('================================\n')
    with open(PATH, 'a') as f:
        f.write('{}\t{}\t{}\t{}\n'.format(i_episode, SR_mean, AvgT_mean, reward_mean))
    return SR_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-seed', type=int, default=1, help='random seed.')
    parser.add_argument('--num_gpus', type=int, default=1, help='number of gpus.')
    parser.add_argument('--epochs', '-me', type=int, default=50000, help='the number of RL train epoch')
    parser.add_argument('--gamma', type=float, default=0.999, help='reward discount factor.')
    parser.add_argument('--learning_rate', type=float, default=1e-6, help='learning rate.')

    parser.add_argument('--data_name', type=str, default='esc', choices=['esc','cima','cb','csa'],
                        help='One of {esc, cima, cb}.')
    parser.add_argument('--system', type=str, default='vicuna', choices=['vicuna','chatgpt','llama2','qwen'],
                        help='One of {vicuna, chatgpt, llama2, qwen}.')
    parser.add_argument('--user', type=str, default='vicuna', choices=['vicuna','chatgpt','llama2','qwen'],
                        help='One of {vicuna, chatgpt, llama2, qwen}.')
    parser.add_argument('--critic', type=str, default='vicuna', choices=['vicuna','chatgpt','llama2','qwen'],
                        help='One of {vicuna, chatgpt, llama2, qwen}.')
    parser.add_argument('--sft_dir', default='sft', #../pretrain/outputs/best_pretrain.pt
                        type=str, help="Pretrain model path.")
    parser.add_argument('--max_turn', type=int, default=8, help='max conversation turn')
    parser.add_argument('--mode', type=str, default='train', help='the mode in [train, test]')
    parser.add_argument('--load_rl_epoch', type=int, default=0, help='load agent from epoch')
    parser.add_argument('--resume', action='store_true',
                        help='Continue an interrupted run of this exact config instead '
                             'of restarting it. Auto-detects the highest RL-agent '
                             'checkpoint already saved under this run\'s filename '
                             '(same data_name/sft_dir/system/user/critic/csa_reward/'
                             'seed as before) and loads it as if --load_rl_epoch had '
                             'named it, unless --load_rl_epoch was already given '
                             'explicitly. Training step numbering continues from '
                             'there rather than restarting at 1. The pre-training '
                             '(epoch-0) evaluation pass is skipped if its output file '
                             'is already on disk. A no-op if nothing to resume from.')


    parser.add_argument("--cache_dir", default='/storage_fast/ydeng/plm', type=str, help="The cache directory.")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--model_path", type=str, default="/storage_fast/ydeng/llm/vicuna_hf/7B")
    parser.add_argument("--model_name", type=str, default="roberta")
    parser.add_argument("--model_name_or_path", default='/scratch/rohank__iitp/roberta-large', type=str, help="model name or path")

    parser.add_argument("--do_lower_case", action='store_false', help="Set this flag if you are using an uncased model.")

    parser.add_argument('--max_steps', type=int, default=10, help='max training steps')
    parser.add_argument('--sample_times', type=int, default=100, help='the epoch of sampling')
    parser.add_argument('--rollout_batch', type=int, default=1,
                        help='Episodes to roll out in parallel per training step, '
                             'sharing one padded generate() call per turn across the '
                             'batch instead of one call per episode. Only implemented '
                             'for --data_name csa with --system/--user/--critic all '
                             'qwen (either --csa_reward arm); falls back to the '
                             'sequential one-episode-at-a-time loop otherwise. Each '
                             'episode still gets its own independent policy-gradient '
                             'update with unchanged math -- this only parallelizes the '
                             'rollout, not the learning. --csa_reward critic additionally '
                             'batches its ten-sample LLM-judge draw across episodes '
                             '(_qwen_critic_reward_batch), on top of the per-turn '
                             'batching. Bounded by GPU memory: the KV cache scales with '
                             'rollout_batch x sequence length.')
    parser.add_argument('--eval_batch', type=int, default=0,
                        help='Test cases to score in parallel per evaluate() call, same '
                             'mechanism as --rollout_batch but for the test set. 0 (the '
                             'default) reuses --rollout_batch. Only takes effect under '
                             'the same conditions as batched training rollout: '
                             '--data_name csa, --system/--user/--critic all qwen, and '
                             '--rollout_batch > 1 -- otherwise evaluate() falls back to '
                             'the sequential one-case-at-a-time loop.')
    parser.add_argument('--eval_num', type=int, default=1, help='the number of steps to evaluate RL model and metric')
    parser.add_argument('--save_num', type=int, default=1, help='the number of steps to save RL model and metric')
    parser.add_argument('--keep_ckpt', type=int, default=1,
                        help='How many RL checkpoints to retain. Each is ~1.05 GB; the '
                             'disk here has only a few GB spare and a full disk kills '
                             'torch.save mid-write and ends the run.')


    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval.")

    # local qwen backend
    parser.add_argument('--qwen_path', default='/scratch/rohank__iitp/Qwen3-8B')
    parser.add_argument('--qwen_dtype', default='bfloat16',
                        choices=['bfloat16', 'float16', 'float32'])
    parser.add_argument('--qwen_device_map', default='cuda:0')
    parser.add_argument('--qwen_max_input_tokens', type=int, default=3072,
                        help='Left-truncate the Qwen chat prompt (system/user/critic '
                             'generation calls) to this many tokens before generate(). '
                             'Conversations grow every turn, so without a cap the '
                             'prefill cost grows with rollout_batch x turn number at '
                             'once, which is what runs a 40GB card out of memory well '
                             'before rollout_batch itself looks large. 0 disables '
                             'truncation and restores the old unbounded behaviour.')
    # API backend. The key is read from OPENAI_API_KEY, never from source.
    parser.add_argument('--openai_model', default='gpt-3.5-turbo-0613',
                        help="Model id for the chatgpt backend. The paper pins "
                             "gpt-3.5-turbo-0613, which is a retired snapshot.")

    # csa. Defaults reproduce upstream behaviour wherever a choice exists.
    parser.add_argument('--csa_reward', default='critic', choices=['critic', 'verifier'],
                        help="'critic' is upstream PPDPP: an LLM judge supplies the "
                             "reward. 'verifier' scores the dataset's executable checks.")
    parser.add_argument('--critic_every', type=int, default=1)
    parser.add_argument('--critic_batch', type=int, default=10,
                        help='Critic samples per forward pass. 10 is upstream; lowering '
                             'it is distributionally identical and cuts peak memory.')
    parser.add_argument('--csa_reveal_threshold', type=float, default=0.35)
    parser.add_argument('--csa_settlement_max_tokens', type=int, default=512)
    # terminal reward weights
    parser.add_argument('--csa_w_content', type=float, default=0.5)
    parser.add_argument('--csa_w_prov', type=float, default=0.5)
    parser.add_argument('--csa_w_halluc', type=float, default=0.0,
                        help='Penalty on crediting facts never disclosed.')
    parser.add_argument('--csa_w_schema', type=float, default=0.0)
    parser.add_argument('--csa_w_accept', type=float, default=0.0,
                        help='Judged acceptance conditions. >0 reintroduces an LLM judge.')
    # potential-based shaping weights (all free -- the detectors are lexical)
    parser.add_argument('--csa_w_shape_disc', type=float, default=1.0)
    parser.add_argument('--csa_w_shape_elic', type=float, default=0.0)
    parser.add_argument('--csa_w_shape_cover', type=float, default=0.3,
                        help='Coverage is gameable in the terminal reward but safe in '
                             'the potential, where shaping provably cannot change the '
                             'optimum. It is also the only potential that varies.')
    # multi-dimensional terminal reward: pool / use / close, plus an integrity gate
    parser.add_argument('--csa_w_use', type=float, default=0.5,
                        help='Weight on dca -- decisive checks passed. Primary axis.')
    parser.add_argument('--csa_w_pool', type=float, default=0.3,
                        help='Weight on disclosure_rate -- private facts surfaced.')
    parser.add_argument('--csa_w_close', type=float, default=0.2,
                        help='Weight on content checks no decisive fact flips.')
    parser.add_argument('--csa_w_halluc_pen', type=float, default=0.5,
                        help='Subtractive penalty on crediting undisclosed facts.')
    parser.add_argument('--csa_done_tau', type=float, default=0.6,
                        help='Success threshold on dca. Five checks in the corpus are '
                             'unreachable, so 1.0 is not attainable on every case.')
    parser.add_argument('--csa_leak_invalidates', type=int, default=1,
                        help='An agent stating a fact it never saw voids the episode.')
    parser.add_argument('--csa_resolve_provenance', type=int, default=1,
                        help='Resolve fact ids from settlement content instead of '
                             'requiring the chair to emit labels it never sees.')

    add_model_args(parser)
    args = parser.parse_args()
    
    #os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    #args.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    print(args.device)
    print('data_set:{}'.format(args.data_name))

    dataset = load_dataset(args.data_name)

    if args.data_name == 'csa':
        # SR_turn is a flat list of length max_turn, so a cap below the longest scenario
        # silently truncates the SR@t curve rather than erroring.
        longest = max(c['interaction_config']['turn_cap'] for c in dataset['test'])
        if args.max_turn < longest:
            print('WARNING: --max_turn %d is below the longest scenario cap %d; raising '
                  'it so the SR@t curve is not truncated.' % (args.max_turn, longest))
            args.max_turn = longest

    filename = '{}-{}-{}-{}-{}'.format(args.data_name,args.sft_dir,args.system,args.user,args.critic)
    if args.data_name == 'csa':
        # Seed and reward mode too: upstream derives every output path from backbone and
        # sft_dir alone, so two seeds -- or the two arms -- would overwrite each other's
        # checkpoints and append to the same eval files.
        filename = '{}-{}-seed{}'.format(filename, args.csa_reward, args.seed)

    config = cfg[args.model_name].from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    tokenizer = tok[args.model_name].from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case, cache_dir=args.cache_dir)

    if args.sft_dir:
        args.sft_dir = os.path.join(args.sft_dir, args.data_name, args.model_name, 'best_checkpoint')
    if not os.path.exists(args.sft_dir):
        print("no sft model, randomly initialize policy model")
        args.sft_dir = None

    train(args, config, dataset, filename, tokenizer)

if __name__ == '__main__':
    main()