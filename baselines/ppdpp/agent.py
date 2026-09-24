from torch.distributions import Categorical
import contextlib
import random
import numpy as np
from transformers import BertModel, RobertaModel, AutoTokenizer
try:
    from transformers import AdamW
except ImportError:
    # transformers removed its own AdamW. torch's differs in two defaults that would
    # silently change training (eps 1e-8 vs 1e-6, weight_decay 1e-2 vs 0), so the
    # HuggingFace values are restored and the call sites stay as upstream wrote them.
    from torch.optim import AdamW as _TorchAdamW

    def AdamW(params, lr=1e-3, **kw):
        kw.setdefault('eps', 1e-6)
        kw.setdefault('weight_decay', 0.0)
        return _TorchAdamW(params, lr=lr, **kw)
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from utils import *
from prompt import ESConvAct, CIMAAct, CBAct, CSAAct

model = {'bert': BertModel, 'roberta': RobertaModel}
act = {'esc': ESConvAct, 'cima': CIMAAct, 'cb': CBAct, 'csa': CSAAct}
TMP_DIR = {
    'esc': './tmp/esc',
    'cima': './tmp/cima',
    'cb': './tmp/cb',
    'csa': './tmp/csa',
}

class PPDPP(nn.Module):
    def __init__(self, args, config, tokenizer):
        super().__init__()
        self.policy = model[args.model_name].from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config, cache_dir=args.cache_dir)
        self.dropout = nn.Dropout(0.5)
        self.act = sorted(list(act[args.data_name].keys()))
        self.classifier = nn.Linear(config.hidden_size, len(self.act))
        self.tokenizer = tokenizer
        self.optimizer = AdamW(
            self.parameters(), lr=args.learning_rate
        )
        self.eps = np.finfo(np.float32).eps.item()
        self.config = config
        self.args = args
        self.saved_log_probs = []
        self.rewards = []
        # Optional per-class loss weights, set by sft.py --class_weights. Left as None
        # the loss is plain unweighted CE, exactly as upstream.
        self.class_weights = None
        # Upstream never placed the policy network on a device, so it silently ran on
        # CPU -- forward+backward through roberta-large, once per dialogue turn, every
        # episode -- while the GPU sat mostly idle running only the backbone LLM. Moving
        # it here (in-place: same Parameter objects, so the AdamW state built above stays
        # valid) is the single biggest speedup available without touching the rollout
        # structure.
        self.to(args.device)

    def build_input(self, state):
        dial_id = []
        for turn in state[::-1]:
            s = self.tokenizer.encode("%s: %s" % (turn['role'], turn['content']))
            if len(dial_id) + len(s) > self.args.max_seq_length:
                break
            dial_id = s[1:] + dial_id
        inp = s[:1] + dial_id
        return [inp]

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.policy(input_ids=input_ids, attention_mask=attention_mask)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        if labels is not None:
            w = None
            if self.class_weights is not None:
                w = self.class_weights.to(logits.device, dtype=logits.dtype)
            loss_fct = CrossEntropyLoss(weight=w)
            loss = loss_fct(logits.view(-1, len(self.act)), labels.view(-1))
            return loss
        else:
            return F.softmax(logits, dim=-1)

    def select_action(self, state, is_test=False):
        inp = self.build_input(state)
        # Same device as self.policy (set in __init__). Building a CPU tensor and
        # forwarding it through a GPU module would error; forwarding a GPU tensor
        # through a CPU module was the original bug -- both are avoided by matching
        # whatever device the network actually lives on.
        device = next(self.policy.parameters()).device
        inp = torch.tensor(inp, device=device).long()

        if is_test:
            # Eval-time selection never backprops, so building the autograd graph here
            # was pure waste -- extra memory and compute on every evaluation episode.
            with torch.no_grad():
                outputs = self.policy(inp)
                pooled_output = outputs[1]
                pooled_output = self.dropout(pooled_output)
                logits = self.classifier(pooled_output)
                probs = nn.functional.softmax(logits, dim=1)
            action = probs.argmax().item()
            return self.act[action]

        outputs = self.policy(inp)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        probs = nn.functional.softmax(logits, dim=1)
        m = Categorical(probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))
        return self.act[action]

    def select_action_batch(self, states, is_test=False):
        """Batched sibling of select_action: one forward pass across all `states`
        instead of one per state. Right-padding is fine here (unlike the Qwen batched
        generation) because Roberta's pooled_output is the representation of the first
        token, which sits at the same position in every row regardless of padding on
        the right; the attention_mask still keeps the pad positions from leaking into
        that representation through self-attention.

        Runs entirely under no_grad, in both modes -- this is rollout only (picking an
        action to drive the environment forward), never learning. A batched forward
        pass shares ONE computation graph across every row in it; if that graph's
        log_probs were later split apart and handed to separate per-episode
        backward()/optimizer.step() calls (as an earlier version of this did), the
        second episode to touch a shared tick's graph would crash with "Trying to
        backward through the graph a second time" -- PyTorch frees a graph's buffers
        after its first backward(), and every episode active in the same tick shares
        that one graph. Learning instead happens afterward via logprob_of_action,
        which redoes each episode's forward pass alone (see its docstring).

        Returns (actions, actions_idx) in training mode -- actions_idx is a detached
        LongTensor aligned with `states`, one entry per row, for the caller to record
        alongside the state that produced it and replay through logprob_of_action once
        the episode is done. In eval mode (is_test=True) returns just the actions list.
        """
        device = next(self.policy.parameters()).device
        encoded = [self.build_input(s)[0] for s in states]
        maxlen = max(len(e) for e in encoded)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id or 0
        input_ids = torch.full((len(encoded), maxlen), pad_id, dtype=torch.long,
                               device=device)
        attention_mask = torch.zeros((len(encoded), maxlen), dtype=torch.long,
                                     device=device)
        for i, e in enumerate(encoded):
            input_ids[i, :len(e)] = torch.tensor(e, dtype=torch.long, device=device)
            attention_mask[i, :len(e)] = 1

        with torch.no_grad():
            outputs = self.policy(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = self.dropout(outputs[1])
            logits = self.classifier(pooled_output)
            probs = nn.functional.softmax(logits, dim=1)

        if is_test:
            actions_idx = probs.argmax(dim=1)
            return [self.act[a.item()] for a in actions_idx]

        m = Categorical(probs)
        actions_idx = m.sample()
        return [self.act[a.item()] for a in actions_idx], actions_idx.detach()

    def logprob_of_action(self, state, action_idx):
        """Recompute, WITH gradient tracking, the log-probability of an action that
        was already chosen during batched (no-grad) rollout selection.

        This is what lets batched rollout give each episode a fully independent
        forward/backward/optimizer-step -- exactly the sequential path's math, one
        episode at a time -- without ever sharing an autograd graph node across
        episodes the way one shared batched forward call would. The extra forward pass
        this costs is a single small Roberta call; negligible next to the Qwen
        generation calls that dominate wall-clock time.
        """
        inp = self.build_input(state)
        device = next(self.policy.parameters()).device
        inp = torch.tensor(inp, device=device).long()
        outputs = self.policy(inp)
        pooled_output = self.dropout(outputs[1])
        logits = self.classifier(pooled_output)
        probs = nn.functional.softmax(logits, dim=1)
        m = Categorical(probs)
        return m.log_prob(action_idx.to(device).view(1)).squeeze(0)

    def optimize_model(self):
        R = 0
        policy_loss = []
        rewards = []
        for r in self.rewards[::-1]:
            # self.rewards elements arrive as single-element GPU tensors (run.py puts
            # them on args.device). Reduce to a plain float here so accumulation doesn't
            # silently build a chain of tiny GPU ops, and so the tensor built below lands
            # on the right device instead of defaulting to CPU.
            R = float(r) + self.args.gamma * R
            rewards.insert(0, R)
        # log_prob (from select_action) now lives on the policy's device since the fix
        # above; rewards must match or the multiply below raises a device mismatch.
        device = next(self.policy.parameters()).device
        rewards = torch.tensor(rewards, device=device)

        # Watch the RAW rewards, not the discounted returns.
        #
        # A constant per-step reward does NOT produce zero advantage: discounting makes
        # R_t vary with position anyway. r = -0.1 with gamma = 0.999 over four turns gives
        # returns [-0.399, -0.300, -0.200, -0.100], std 0.129, which whitens to roughly
        # [-1.34, -0.45, +0.45, +1.34].
        #
        # That is worse than no gradient, because it looks like one. The pattern is purely
        # positional -- it says "prefer whatever you did late in the episode" -- and is
        # identical whatever the policy chose. The 1000-episode verifier-arm run trained
        # against exactly this: disclosure was 0.0 on 91% of steps, potential-based shaping
        # telescopes to zero over an episode, so the reward was the constant -0.1 and every
        # update pushed on turn position rather than on anything the planner did.
        #
        # So the test is whether the reward DISCRIMINATES between turns, before discounting
        # smears position into it.
        self.degenerate_updates = getattr(self, 'degenerate_updates', 0)
        self.total_updates = getattr(self, 'total_updates', 0) + 1
        raw = [float(r) for r in self.rewards]
        if len(raw) > 1 and (max(raw) - min(raw)) < 1e-8:
            self.degenerate_updates += 1
            if self.degenerate_updates in (1, 10, 100) or \
                    self.degenerate_updates % 500 == 0:
                print('[WARN] constant reward across this episode (r=%.4f): the advantage '
                      'that follows is positional, not behavioural, and this update '
                      'teaches turn order rather than policy. %d/%d updates so far. '
                      'Check --csa_reward and the shaping weights.'
                      % (raw[0], self.degenerate_updates, self.total_updates), flush=True)
        if rewards.shape[0] > 1:
            rewards = (rewards - rewards.mean()) / (rewards.std() + self.eps)

        for log_prob, reward in zip(self.saved_log_probs, rewards):
            policy_loss.append(-log_prob * reward)
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        del self.rewards[:]
        del self.saved_log_probs[:]
        return policy_loss.data

    def optimize_from_buffer(self, log_probs, rewards):
        """Same REINFORCE update as optimize_model -- identical discounting, identical
        return-whitening, identical degenerate-reward bookkeeping -- but taking the
        per-episode log_prob/reward lists as explicit arguments instead of reading
        self.saved_log_probs/self.rewards.

        This is what makes batched rollout safe: a batch of episodes finishes out of
        lockstep (different turn_order / utterance_cap per case), so there's no single
        shared trajectory buffer the way the sequential loop has one. Each episode's
        buffer is collected separately by the caller (run.py's batched training loop)
        and handed here for its own, independent optimizer step -- so this produces
        exactly the same number of gradient updates, with exactly the same per-episode
        math, as running the same episodes through the sequential path one at a time.
        Only the rollout (the LLM calls that generate the trajectory) is batched, not
        the learning.
        """
        raw = [float(r) for r in rewards]
        R = 0.0
        returns = []
        for r in raw[::-1]:
            R = r + self.args.gamma * R
            returns.insert(0, R)

        self.degenerate_updates = getattr(self, 'degenerate_updates', 0)
        self.total_updates = getattr(self, 'total_updates', 0) + 1
        if len(raw) > 1 and (max(raw) - min(raw)) < 1e-8:
            self.degenerate_updates += 1
            if self.degenerate_updates in (1, 10, 100) or \
                    self.degenerate_updates % 500 == 0:
                print('[WARN] constant reward across this episode (r=%.4f): the advantage '
                      'that follows is positional, not behavioural, and this update '
                      'teaches turn order rather than policy. %d/%d updates so far. '
                      'Check --csa_reward and the shaping weights.'
                      % (raw[0], self.degenerate_updates, self.total_updates), flush=True)

        device = next(self.policy.parameters()).device
        returns_t = torch.tensor(returns, device=device)
        if returns_t.shape[0] > 1:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + self.eps)

        policy_loss = [-log_prob * reward
                       for log_prob, reward in zip(log_probs, returns_t)]
        self.optimizer.zero_grad()
        # log_probs here are 0-dim scalars (one per turn, from logprob_of_action's
        # single-episode replay), unlike optimize_model's 1-element tensors, so stack
        # instead of cat.
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        return policy_loss.data

    def degenerate_fraction(self):
        """Share of updates that produced exactly zero advantage. Report this next to any
        learning curve: a flat curve with a high fraction here is a broken reward, not a
        limitation of the planner."""
        tot = getattr(self, 'total_updates', 0)
        return (getattr(self, 'degenerate_updates', 0) / tot) if tot else 0.0
    
    def save_model(self, data_name, filename, epoch_user):
        output_dir = TMP_DIR[data_name] + '/RL-agent/' + filename + '-epoch-{}'.format(epoch_user)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        torch.save(self.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))
        torch.save(self.args, os.path.join(output_dir, 'training_args.bin'))
    def load_model(self, data_name, filename, epoch_user=None):
        if epoch_user: 
            output_dir = TMP_DIR[data_name] + '/RL-agent/' + filename + '-epoch-{}'.format(epoch_user)
        else:
            output_dir = filename
        if hasattr(self, 'module'):
            self.module.load_state_dict(torch.load(os.path.join(output_dir, 'pytorch_model.bin')))
        else:
            self.load_state_dict(torch.load(os.path.join(output_dir, 'pytorch_model.bin'), map_location='cuda:0'))