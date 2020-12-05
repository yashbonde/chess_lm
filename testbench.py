import wandb
wandb.init(project="sweep-practice")
from transformers import GPT2Model, GPT2Config as ModelConfig
import transformers
from torch import nn
from torch.nn import Conv2d, BatchNorm2d, ReLU, Linear, Tanh
import torch

from argparse import ArgumentParser

from tqdm import trange
import numpy as np
import random

from model import BetaChess, ModelConfig, TrainerConfig, PolicyHead, MLP, F, configure_optimizers, init_weights

args = ArgumentParser()
args.add_argument("--scheduler", type=str)
args.add_argument("--warmup_perc", type=float)
args.add_argument("--weight_decay", type=float)
args = args.parse_args()

VOCAB = 39
MAXLEN = 10

def set_seed(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

set_seed(4)

def fetch(url):
    import requests
    import os
    import hashlib
    import tempfile
    fp = os.path.join(tempfile.gettempdir(), hashlib.md5(
        url.encode('utf-8')).hexdigest())
    if os.path.isfile(fp) and os.stat(fp).st_size > 0:
        with open(fp, "rb") as f:
            dat = f.read()
    else:
        print("fetching %s" % url)
        dat = requests.get(url).content
        with open(fp+".tmp", "wb") as f:
            f.write(dat)
        os.rename(fp+".tmp", fp)
    return dat

def create_datasets(n, r = False, c = False):
    if r and c: # like our dataset
        data = fetch("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt").decode("utf-8").lower()
        chars = sorted(list(set(data)))
        vocab = {ch: i for i, ch in enumerate(chars)}
        inv_vocab = {i: ch for i, ch in enumerate(chars)}
        maxlen = MAXLEN
        vocab_size = len(chars)
        data = np.array([vocab[x] for x in data][:maxlen * n]).reshape(-1, maxlen)
        x = data * np.pi / 180
        y = np.sin(x)
        return data[:n], (y[:n] - 0.25) * 2

    elif r: # simple regression --> sin + noise
        input_ = np.arange(VOCAB).tolist() * n
        input_ = np.array(input_[:-(len(input_) % (MAXLEN))]).reshape(-1, MAXLEN)
        x = np.pi * input_
        y = np.sin(x)
        y = y.reshape(-1, MAXLEN)
        return input_[:n], (y[:n] - 0.25) * 4
    
    elif c: # simple classification
        data = fetch("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt").decode("utf-8").lower()
        chars = sorted(list(set(data)))
        vocab = {ch: i for i, ch in enumerate(chars)}
        inv_vocab = {i: ch for i, ch in enumerate(chars)}
        maxlen = MAXLEN
        data = np.array([vocab[x] for x in data][:maxlen * n]).reshape(-1, maxlen)
        return data, inv_vocab

####### MODEL ########
class BetaChess(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        config = ModelConfig(**vars(self.config))
        config.n_layer = config.n_layer - 1
        self.body = GPT2Model(config) # residual tower in AlphaZero
        
        # the policy head and value head are now similar to what is in AlphaZero
        self.policy_head = PolicyHead(config)
        self.value_head = nn.Sequential(*[
            MLP(config.n_embd, config),
            nn.ReLU(),
            nn.Linear(config.n_embd, 1),
            nn.Tanh()
        ])

    def forward(self, input_ids, value_targets = None, loss = None, **gptkwargs):
        x = self.body(input_ids, return_dict = True, **gptkwargs)
        logits = self.policy_head(x.last_hidden_state)
        values = self.value_head(x.last_hidden_state)
        out = (logits, values)
        if loss is not None and value_targets is not None:            
            # Categorical cross entropy loss worked best for policy
            logits_reshape = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
            targets = input_ids[:, 1:].contiguous().view(-1)
            loss_policy = F.cross_entropy(logits_reshape, targets)

            # MSE works best for value function
            loss_value = (values[:, :-1].contiguous().view(-1) - value_targets[:,1:].contiguous().view(-1)) ** 2
            loss_value = loss_value.mean()

            loss = loss_policy + loss_value

            out = (logits, values, (loss, loss_policy, loss_value))
        return out



# ---- config ---- #
modelConf = ModelConfig(
    vocab_size = VOCAB,
    n_positions=MAXLEN,
    n_ctx=MAXLEN,
    n_embd=2,
    n_layer=2,
    n_head=1,
    activation_function="relu",
    resid_pdrop=0.0,
    embd_pdrop=0.0,
    attn_pdrop=0.0,
)
model = BetaChess(modelConf)
print(modelConf)
print(f"Model Size: {sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())}")
model.apply(init_weights)

trainerConf = TrainerConfig(
    num_epochs = 1,
    batch_size = 3,
    lr = 0.0001,
    betas = (0.9, 0.98),
    tb_path = "models/",
    test_every = 10,
    ckpt_path = "models",
    patience = 1,
    scheduler=args.scheduler,
    t0div = 5,
    tmult = 2,
    warmup_perc =args.warmup_perc,
    weight_decay =args.weight_decay,
)
optim = configure_optimizers(model, trainerConf)

# --- train --- #
wandb.init(config={"scheduler": "WarmupConstant", "warmup_perc":0.14, "weight_decay":0.1})
X, Y = create_datasets(n = 40, c= True, r = True)
pbar = trange(500)
for i in pbar:
    logits, values, (loss, loss_policy, loss_value) = model(
        input_ids=torch.Tensor(X).long(), value_targets = torch.Tensor(Y),
        loss = True
    )
    loss.backward()
    optim.step()
    pbar.set_description(f"{i}, {loss:.3f} ({loss_policy:.3f}, {loss_value:.3f})")

    wandb.log({"Loss": loss})
