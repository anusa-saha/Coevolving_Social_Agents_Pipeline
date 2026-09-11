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



def train(args, config, dataset, filename, tokenizer):
    env = Env(args, dataset, mode='train') # env init
    set_random_seed(args.seed)
    policy = PPDPP(args, config, tokenizer) # policy network init

    # load policy parameters
    if args.sft_dir is not None:
        print('Staring loading policy model from {}'.format(args.sft_dir))
        policy.load_model(data_name=args.data_name, filename=args.sft_dir)
    
    if args.load_rl_epoch > 0:
        print('Staring loading rl model in epoch {}'.format(args.load_rl_epoch))
        policy.load_model(data_name=args.data_name, filename=filename, epoch_user=args.load_rl_epoch)
    

    test_performance = []
    if args.do_eval:
        SR15_mean = evaluate(args, dataset, policy, filename, 0, env)
        test_performance = [SR15_mean]
    if not args.do_train:
        return
    for train_step in range(1, args.max_steps+1):
        SR, AvgT, total_reward = 0., 0., 0.
        loss = torch.tensor(0, dtype=torch.float, device=args.device)
        for i_episode in tqdm(range(args.sample_times),desc='sampling'):
            #blockPrint()
            print('\n================new tuple:{}===================='.format(i_episode))
            state = env.reset()

            epi_reward = 0
            done = False
            for t in count():   # user  dialog
                action = policy.select_action(state)
                state, reward, done = env.step(action)
                epi_reward += reward
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
                dirs = [os.path.join(ck_root, d) for d in os.listdir(ck_root)]
                dirs = sorted([d for d in dirs if os.path.isdir(d)],
                              key=os.path.getmtime, reverse=True)
                for old in dirs[max(1, args.keep_ckpt):]:
                    shutil.rmtree(old, ignore_errors=True)
                    print('[janitor] removed old checkpoint %s' % os.path.basename(old))
            except Exception as e:
                print('[janitor] skipped: %s' % e)
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
                    # Everything the metric catalogue needs, so any number can be
                    # recomputed offline without re-running a model.
                    record.update({
                        'uid': test_env.case.get('uid'),
                        'domain': test_env.case.get('domain'),
                        'num_agents': test_env.case.get('num_agents'),
                        'scenario_type': test_env.case.get('scenario_type'),
                        'settlement': test_env.settlement,
                        'score': test_env.last_score,
                        'score_norm': test_env.last_score_norm,
                        'floor': floor_score(test_env.case),
                        'revealed': sorted(test_env.revealed),
                        'reveal_elicited': dict(test_env.reveal_elicited),
                        'addressed': sorted(test_env.addressed),
                        'leaks': list(test_env.leaks),
                        'done': done,
                        # --- efficiency and cost (catalogue section C) ---
                        'turns': t + 1,
                        'max_turn': test_env.max_turn,
                        'n_calls': getattr(test_env, 'n_calls', 0),
                        'calls_by_role': dict(getattr(test_env, 'calls_by_role', {})),
                        'prompt_chars': getattr(test_env, 'prompt_chars', 0),
                        # --- policy diagnostics (catalogue section G) ---
                        'act_history': list(getattr(test_env, 'act_history', [])),
                        'reveal_turn': dict(getattr(test_env, 'reveal_turn', {})),
                    })
                rec_file.write('%s\n\n' % str(record))
                break

        enablePrint()
            
    
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


    parser.add_argument("--cache_dir", default='/storage_fast/ydeng/plm', type=str, help="The cache directory.")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--model_path", type=str, default="/storage_fast/ydeng/llm/vicuna_hf/7B")
    parser.add_argument("--model_name", type=str, default="roberta")
    parser.add_argument("--model_name_or_path", default='roberta-large', type=str, help="model name or path")

    parser.add_argument("--do_lower_case", action='store_false', help="Set this flag if you are using an uncased model.")

    parser.add_argument('--max_steps', type=int, default=10, help='max training steps')
    parser.add_argument('--sample_times', type=int, default=100, help='the epoch of sampling')
    parser.add_argument('--eval_num', type=int, default=1, help='the number of steps to evaluate RL model and metric')
    parser.add_argument('--save_num', type=int, default=1, help='the number of steps to save RL model and metric')
    parser.add_argument('--keep_ckpt', type=int, default=1,
                        help='How many RL checkpoints to retain. Each is ~1.05 GB; the '
                             'disk here has only a few GB spare and a full disk kills '
                             'torch.save mid-write and ends the run.')


    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval.")

    # local qwen backend
    parser.add_argument('--qwen_path', default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--qwen_dtype', default='bfloat16',
                        choices=['bfloat16', 'float16', 'float32'])
    parser.add_argument('--qwen_device_map', default='cuda:0')
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