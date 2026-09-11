from torch.distributions import Categorical
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
        inp = torch.tensor(inp).long()

        outputs = self.policy(inp)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        probs = nn.functional.softmax(logits, dim=1)
        m = Categorical(probs)
        if is_test:
            action = probs.argmax().item()
        else:
            action = m.sample()
            self.saved_log_probs.append(m.log_prob(action))
        return self.act[action]

    def optimize_model(self):
        R = 0
        policy_loss = []
        rewards = []
        for r in self.rewards[::-1]:
            R = r + self.args.gamma * R
            rewards.insert(0, R)
        rewards = torch.tensor(rewards)

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

