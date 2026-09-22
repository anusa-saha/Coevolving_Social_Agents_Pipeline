"""Before a multi-day run: can this box run every stage?

    python parallel/preflight.py --prefetch     # CPU: versions, kernels, GPUs, data, downloads
    python parallel/preflight.py --probe        # one GPU: memory, speed, loss equivalence

run_all.py runs both as its first two jobs, so a missing kernel, a model that does not fit
a 24 GB card, or a training loss computed wrongly stops the schedule within minutes rather
than a day in.

--probe measures, on the real model:
  * weights on the card, and generation speed with the KV cache and thinking off
  * a LoRA forward+backward with gradient checkpointing at 2,560 tokens -- longer than any
    training sequence here (GRPO scores a settling turn: a long prompt plus 512 tokens)
  * that every LoRA target the arms configure matches a module of this architecture
  * that compat.completion_nll equals the model's own loss; it computes the same number
    from the completion's logits only, which is what makes the backward fit
and exits non-zero if any of them fail.
"""
import argparse
import collections
import importlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Where each arm names its dialogue model. Read as text: importing some of these configs
# resolves paths and data, which is not what a preflight should depend on.
CONFIGS = ('epo/config.py', 'sotopia_rl/config.py', 'sotopia_tom/config.py',
           'sotopia_omega/config.py', 'roundtable/config.py', 'ppdpp/run.py')
PLANNER = 'roberta-large'                        # PPDPP's act classifier
MIN_MIB = 22000
PARA = ('The chair asks each advisor in turn for the one constraint only they know, '
        'checks it against the schedule, and records the settlement. ')


def say(tag, msg):
    print('  %-6s %s' % (tag, msg), flush=True)


def gib(b):
    return b / 2 ** 30


def detect_gpus():
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total',
                              '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return []
    rows = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(',')]
        if len(parts) == 3 and parts[0].isdigit():
            rows.append((int(parts[0]), parts[1], int(float(parts[2]))))
    return rows


def configured_models():
    found = collections.OrderedDict()
    for rel in CONFIGS:
        with open(os.path.join(ROOT, rel), encoding='utf-8') as f:
            for m in re.findall(r"'(Qwen/[^']+)'", f.read()):
                found.setdefault(m, []).append(rel)
    return found


def lora_settings():
    """Sotopia-RL's LoRA settings; EPO and Omega configure the same targets."""
    spec = importlib.util.spec_from_file_location(
        'sr_config', os.path.join(ROOT, 'sotopia_rl', 'config.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    d = mod.Defaults
    return tuple(d.lora_targets), d.lora_r, d.lora_alpha, d.lora_dropout


def write_report(path, section, data):
    report = {}
    if os.path.isfile(path):
        try:
            with open(path, encoding='utf-8') as f:
                report = json.load(f)
        except ValueError:
            report = {}
    report[section] = dict(data, time=time.strftime('%Y-%m-%d %H:%M:%S'))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=1)


# ====================================================================== prefetch
def prefetch(cli):
    problems = []
    print('versions')
    from csa_core import compat
    for k, v in compat.versions().items():
        say('', '%-14s %s' % (k, v or 'MISSING'))
    for pr in compat.problems():
        say('BLOCK', pr)
        problems.append(pr)
    for mod, who in (('huggingface_hub', 'downloads'), ('tensorboardX', 'ppdpp/sft.py'),
                     ('sklearn', 'ppdpp/sft.py'), ('tqdm', 'ppdpp')):
        try:
            importlib.import_module(mod)
        except Exception:                            # noqa: BLE001
            say('BLOCK', '%s is not importable (needed by %s)' % (mod, who))
            problems.append(mod)

    print('\nlinear-attention kernels (Qwen3.5: 24 of its 32 layers)')
    kernels = {}
    for mod, pkg in (('fla', 'flash-linear-attention'), ('causal_conv1d', 'causal-conv1d')):
        try:
            importlib.import_module(mod)
            kernels[pkg] = True
            say('ok', pkg)
        except Exception as e:                       # noqa: BLE001
            kernels[pkg] = False
            say('SLOW', '%s not importable (%s): transformers falls back to a pure-torch '
                        'path, which is much slower. pip install %s'
                % (pkg, type(e).__name__, pkg))
            if cli.strict:
                problems.append(pkg)

    print('\ngpus')
    gpus = detect_gpus()
    if not gpus:
        say('WARN', 'nvidia-smi lists no GPUs')
    for i, name, mib in gpus:
        say('ok' if mib >= MIN_MIB else 'WARN', 'gpu %d  %s  %d MiB' % (i, name, mib))

    print('\ndata')
    from csa_core import data_csa
    sizes = {k: len(v) for k, v in data_csa.load().items()}
    say('ok', '%s from %s' % (sizes, data_csa.source()))

    print('\nmodels')
    models = configured_models()
    if len(models) != 1:
        say('WARN', 'the arms name different models: %s' % dict(models))
    for m, where in models.items():
        say('', '%s  (%s)' % (m, ', '.join(where)))
    if cli.no_download:
        say('skip', 'downloads (--no_download)')
    else:
        from huggingface_hub import snapshot_download
        skip = ['*.h5', '*.msgpack', '*.ot', 'onnx/*', '*.onnx', 'rust_model*',
                'tf_model*', 'flax_model*', 'coreml/*']
        for repo in list(models) + [PLANNER]:
            t = time.time()
            path = snapshot_download(repo_id=repo, ignore_patterns=skip)
            say('ok', '%s -> %s (%.0fs)' % (repo, path, time.time() - t))
        import nltk
        for pkg in ('punkt', 'punkt_tab'):
            say('ok' if nltk.download(pkg, quiet=True) else 'WARN', 'nltk %s' % pkg)

    write_report(cli.out, 'prefetch', {'problems': problems, 'kernels': kernels,
                                       'gpus': gpus, 'split': sizes,
                                       'models': list(models)})
    if problems:
        print('\nBLOCKING: %s' % '; '.join(problems))
        return 1
    print('\npreflight --prefetch: ok')
    return 0


# ====================================================================== probe
def probe(cli):
    import torch
    from csa_core import compat
    if not torch.cuda.is_available():
        print('no CUDA device visible')
        return 1
    fails, rep = [], {}
    model_id = next(iter(configured_models()), 'Qwen/Qwen3.5-9B')
    targets, r, alpha, dropout = lora_settings()
    dev = 'cuda:0'
    props = torch.cuda.get_device_properties(0)
    total = props.total_memory
    print('probe on %s (%.1f GiB), model %s' % (props.name, gib(total), model_id))

    t = time.time()
    tok = compat.load_tokenizer(model_id)
    model = compat.load_causal_lm(model_id, 'bfloat16', dev, trainable=True)
    torch.cuda.synchronize()
    weights = torch.cuda.memory_allocated(0)
    params = sum(p.numel() for p in model.parameters())
    say('ok', '%s loaded in %.0fs: %.2fB params, %.2f GiB of weights'
        % (type(model).__name__, time.time() - t, params / 1e9, gib(weights)))
    rep.update(model=model_id, arch=type(model).__name__, params_b=round(params / 1e9, 3),
               weights_gib=round(gib(weights), 2), card=props.name,
               card_gib=round(gib(total), 2))

    # ---- generation: KV cache on, thinking off, the way every arm generates
    model.eval()
    msgs = [{'role': 'system', 'content': 'You are chairing a meeting.'},
            {'role': 'user', 'content': PARA * 60 + '\nWhat should the chair ask next?'}]
    enc = tok([compat.render_chat(tok, msgs)], return_tensors='pt').to(dev)
    torch.cuda.reset_peak_memory_stats(0)
    torch.cuda.synchronize()
    t = time.time()
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=cli.gen_tokens,
                             min_new_tokens=cli.gen_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
    torch.cuda.synchronize()
    dt = time.time() - t
    new = out[0][enc['input_ids'].shape[1]:]
    tps = len(new) / max(dt, 1e-9)
    text = tok.decode(new, skip_special_tokens=False)
    say('ok', 'generation: %d prompt + %d new tokens in %.1fs = %.1f tok/s, peak %.2f GiB'
        % (enc['input_ids'].shape[1], len(new), dt, tps,
           gib(torch.cuda.max_memory_allocated(0))))
    say('info', 'an episode generates roughly 1,100 tokens -> about %.0fs each at this rate'
        % (1100 / max(tps, 1e-9)))
    if '<think>' in text:
        say('FAIL', 'the model opened a <think> block: enable_thinking=False is not taking '
                    'effect, and every prompt in the repo assumes plain answers')
        fails.append('thinking not disabled')
    rep.update(gen_tokens_per_s=round(tps, 1),
               gen_peak_gib=round(gib(torch.cuda.max_memory_allocated(0)), 2))
    del out

    # ---- LoRA targets
    from peft import LoraConfig, get_peft_model
    pm = get_peft_model(model, LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout,
                                          bias='none', task_type='CAUSAL_LM',
                                          target_modules=list(targets)))
    hits = collections.Counter(n.rsplit('.', 1)[-1] for n, m in pm.named_modules()
                               if hasattr(m, 'lora_A'))
    missing = [x for x in targets if not hits.get(x)]
    say('FAIL' if missing else 'ok', 'LoRA modules per target: %s%s'
        % (dict(hits), ('   MISSING %s' % missing) if missing else ''))
    if missing:
        fails.append('LoRA targets match nothing: %s' % missing)
    rep['lora_hits'] = dict(hits)

    # ---- one training step at the longest sequence the arms use
    pm.gradient_checkpointing_enable()
    if hasattr(pm, 'enable_input_require_grads'):
        pm.enable_input_require_grads()
    pm.train()
    L, C = cli.train_len, cli.completion_len
    base = tok(PARA * 400, add_special_tokens=False).input_ids
    ids = torch.tensor([(base * (L // len(base) + 1))[:L]], device=dev)
    labels = ids.clone()
    labels[:, :L - C] = -100
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)
    torch.cuda.synchronize()
    t = time.time()
    loss = compat.completion_nll(pm, ids, labels)
    loss.backward()
    torch.cuda.synchronize()
    step = time.time() - t
    peak = torch.cuda.max_memory_allocated(0)
    reserved = torch.cuda.max_memory_reserved(0)
    fits = reserved <= cli.headroom * total
    say('ok' if fits else 'FAIL',
        'LoRA step, %d tokens (%d labelled), checkpointing on: %.1fs, peak %.2f GiB '
        'allocated / %.2f GiB reserved of %.1f GiB'
        % (L, C, step, gib(peak), gib(reserved), gib(total)))
    if not fits:
        fails.append('a training step needs %.1f GiB; the card has %.1f'
                     % (gib(reserved), gib(total)))
    rep.update(train_len=L, train_step_s=round(step, 2), train_peak_gib=round(gib(peak), 2),
               train_reserved_gib=round(gib(reserved), 2))
    pm.zero_grad(set_to_none=True)
    del loss
    torch.cuda.empty_cache()

    # ---- the tail-logit loss must be the model's own loss
    pm.eval()
    with torch.no_grad():
        mine = float(compat.completion_nll(pm, ids, labels))
        theirs = float(pm(input_ids=ids, labels=labels).loss)
    same = abs(mine - theirs) <= 1e-3 * max(1.0, abs(theirs))
    say('ok' if same else 'FAIL', 'completion_nll %.5f vs the model\'s loss %.5f'
        % (mine, theirs))
    if not same:
        fails.append('completion_nll disagrees with the model loss')
    rep.update(completion_nll=mine, model_loss=theirs)

    rep['fails'] = fails
    write_report(cli.out, 'probe', rep)
    if fails:
        print('\nPROBE FAILED: %s' % '; '.join(fails))
        return 1
    print('\npreflight --probe: ok')
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prefetch', action='store_true')
    p.add_argument('--probe', action='store_true')
    p.add_argument('--strict', action='store_true',
                   help='treat missing linear-attention kernels as blocking')
    p.add_argument('--no_download', action='store_true')
    p.add_argument('--train_len', type=int, default=2560)
    p.add_argument('--completion_len', type=int, default=512)
    p.add_argument('--gen_tokens', type=int, default=256)
    p.add_argument('--headroom', type=float, default=0.95,
                   help='fail when a training step reserves more than this share of the card')
    p.add_argument('--out', default=os.path.join(ROOT, 'runs', 'preflight.json'))
    cli = p.parse_args()
    if not (cli.prefetch or cli.probe):
        p.error('pass --prefetch, --probe, or both')
    rc = 0
    if cli.prefetch:
        rc = prefetch(cli) or rc
    if cli.probe and rc == 0:
        rc = probe(cli) or rc
    return rc


if __name__ == '__main__':
    sys.exit(main())
