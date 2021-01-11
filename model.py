"""chess lm model + dataset file"""

import h5py
import json
import time
import math
import wandb
import chess
import pickle
import random
import numpy as np
from tqdm import trange

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import IterableDataset, DataLoader, Dataset

from transformers import PretrainedConfig
from transformers import GPT2Model
from transformers.models.gpt2.modeling_gpt2 import MLP, Attention, Conv1D

# ---- helper function ----- #
def set_seed(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


################################################
####### Model ##################################
################################################

class BaseHFGPT(nn.Module):
    def __init__(self, config, policy_bias = False):
        super().__init__()
        self.config = config
        self.gpt = GPT2Model(config)
        self.policy_head = nn.Linear(config.n_embd, config.vocab_size, bias = policy_bias)

        # final head
        if config.loss_method == "mse":
            self.value_head = nn.Linear(config.n_embd, 1)
        elif config.loss_method == "ce":
            self.value_head = nn.Linear(config.n_embd, 3)

    def forward(self, input_ids, value_targets = None, loss = None, **gptkwargs):
        config = self.config
        x = self.gpt(input_ids, return_dict = True, **gptkwargs)
        logits = self.policy_head(x.last_hidden_state)
        values = self.value_head(x.last_hidden_state)
        out = (logits, values)
        if loss is not None and value_targets is not None:
            # Categorical cross entropy loss worked best for policy
            logits_reshape = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
            targets = input_ids[:, 1:].contiguous().view(-1)
            loss_policy = F.cross_entropy(logits_reshape, targets)

            if config.loss_method == "mse":
                # MSE works best for value function
                values = F.tanh(values)
                loss_value = (values[:, :-1].contiguous().view(-1) - value_targets[:,1:].contiguous().view(-1)) ** 2
                loss_value = loss_value.mean()

            elif config.loss_method == "ce":
                value_reshape = values[:, :-1].contiguous().view(-1, 3)
                value_targets = value_targets[:, 1:].contiguous().view(-1) + 1 # [-1, 0, 1] --> [0, 1, 2]
                loss_value = F.cross_entropy(value_reshape, value_targets.long())

            loss = loss_policy + loss_value

            out = (logits, values, (loss, loss_policy, loss_value))
        return out


class Denorm(torch.nn.Module):
    def __init__(self, normalized_shape, eps = 1e-5):
        super().__init__()
        normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = True
        self.weight = torch.nn.parameter.Parameter(torch.Tensor(*normalized_shape))
        self.bias = torch.nn.parameter.Parameter(torch.Tensor(*normalized_shape))

    def forward(self, x):
        mu = x.mean()
        var = x.var()
        out = (x + mu) * (torch.sqrt(var + self.eps))
        return out * self.weight + self.bias


class PolicyHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # first block
        self.attn = Attention(config.n_embd, config.n_ctx, config, scale = True)
        self.ln = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = MLP(config.n_embd * 4, config)

        # second block
        self.ln2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn2 = Attention(config.n_embd, config.n_ctx, config, scale=True)
        self.ln3 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        # final head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, hidden_states):
        # first block
        attn_output = self.attn(hidden_states)[0]
        hidden_states = attn_output + hidden_states # residual connection
        feed_forward_hidden_states = self.mlp(self.ln(hidden_states))
        hidden_states = hidden_states + feed_forward_hidden_states # residual connection

        # second block
        attn_output = self.attn2(self.ln2(hidden_states))[0]
        hidden_states = self.ln3(attn_output + hidden_states) # residual connection
        out = self.lm_head(hidden_states)
        return out


class ValueHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # first block
        self.attn = Attention(config.n_embd, config.n_ctx, config, scale=True)
        self.ln = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = MLP(config.n_embd * 4, config)

        # second block
        self.ln2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn2 = Attention(config.n_embd, config.n_ctx, config, scale=True)
        self.ln3 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        # final head
        if config.loss_method == "mse":
            self.val_head = nn.Linear(config.n_embd, 1)
        elif config.loss_method == "ce":
            self.val_head = nn.Linear(config.n_embd, 3)

    def forward(self, hidden_states):
        # first block --> complete block
        attn_output = self.attn(hidden_states)[0]
        hidden_states = attn_output + hidden_states # residual connection
        feed_forward_hidden_states = self.mlp(self.ln(hidden_states))
        hidden_states = hidden_states + feed_forward_hidden_states # residual connection

        # second block
        attn_output = self.attn2(self.ln2(hidden_states))[0]
        hidden_states = self.ln3(hidden_states + attn_output) # residual connection
        out =  self.val_head(hidden_states) # value head
        return out


class BetaChess(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        config = ModelConfig(**vars(self.config))
        config.n_layer = config.n_layer - 2
        self.body = GPT2Model(config) # residual tower in AlphaZero

        # the policy head and value head are now similar to what is in AlphaZero
        self.policy_head = PolicyHead(config)
        self.value_head = ValueHead(config)

    def forward(self, input_ids, position_ids = None, value_targets = None, loss = None):
        config = self.config
        x = self.body(input_ids, position_ids = position_ids, return_dict = True)
        logits = self.policy_head(x.last_hidden_state)
        values = self.value_head(x.last_hidden_state)
        out = (logits, values)
        if loss is not None and value_targets is not None:
            # no you stupid categorical cross entropy is not the loss function used for training
            # Categorical cross entropy loss worked best for policy
            logits_reshape = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
            targets = input_ids[:, 1:].contiguous().view(-1)
            loss_policy = F.cross_entropy(logits_reshape, targets)

            # the loss function for policy is loss_policy = -pi.T*log(p)
            # logits_log = F.log_softmax(logits[:, :-1, :].contiguous().view(-1, logits.size(-1)), dim = -1)
            # targets = F.one_hot(targets).float() # convert to one hot encodings
            # loss_policy = -logits_log.T.matmul(targets).mean()

            if config.loss_method == "mse":
                # MSE works best for value function
                values = F.tanh(values)
                loss_value = (values[:, :-1].contiguous().view(-1) - value_targets[:,1:].contiguous().view(-1)) ** 2
                loss_value = loss_value.mean()

            elif config.loss_method == "ce":
                value_reshape = values[:, :-1].contiguous().view(-1, 3)
                value_targets = value_targets[:, 1:].contiguous().view(-1) + 1 # [-1, 0, 1] --> [0, 1, 2]
                loss_value = F.cross_entropy(value_reshape, value_targets.long())

            loss = loss_policy + loss_value # weight regularisation added in optim

            out = (logits, values, (loss, loss_policy, loss_value))
        return out

#
# This method below uses the complete sequence of the game from start to end
# hope is that with this better representation the game play would be better
#
class BetaChessForFullGameSequence(nn.Module):
    # like BetaChess but instead takes the custom labels
    def __init__(self, config):
        super().__init__()
        self.config = config
        config = ModelConfig(**vars(self.config))
        config.n_layer = config.n_layer - 2
        self.body = GPT2Model(config)  # residual tower in AlphaZero

        # the policy head and value head are now similar to what is in AlphaZero
        self.policy_head = PolicyHead(config)
        self.value_head = ValueHead(config)

    def forward(self, input_ids, labels_mask=None, attention_mask=None, value_targets=None, labels=None, loss=None):
        x = self.body(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        logits = self.policy_head(x.last_hidden_state)
        values = self.value_head(x.last_hidden_state)
        out = (logits, values)

        if labels is not None and value_targets is not None:
            if labels_mask is not None:
                # mask illegal moves
                logits = labels_mask + logits

            # categorical cross entropy
            logits_reshape = logits.contiguous().view(-1, logits.size(-1))
            policy_targets = labels.contiguous().view(-1)
            non_pad_idx = policy_targets > 0
            # print(non_pad_idx)
            loss_policy = F.cross_entropy(logits_reshape[non_pad_idx], policy_targets[non_pad_idx])

            # MSE loss
            values_reshape = F.tanh(values).contiguous().view(-1)
            values_targets = value_targets.contiguous().view(-1)
            loss_value = F.mse_loss(values_reshape[non_pad_idx], values_targets[non_pad_idx]).mean()

            loss = loss_policy + loss_value  # weight regularisation added in optim
            out = (logits, values, (loss, loss_policy, loss_value))
        return out


class ValueOnlyNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gpt = GPT2Model(config)

        # final head
        if config.loss_method == "mse":
            self.value_head = nn.Linear(config.n_embd, 1)
        elif config.loss_method == "ce":
            self.value_head = nn.Linear(config.n_embd, 3)

    def forward(self, input_ids, value_targets = None, loss = None, **gptkwargs):
        config = self.config
        x = self.gpt(input_ids, return_dict = True, **gptkwargs)
        values = self.value_head(x.last_hidden_state)
        out = (None, values)
        if loss is not None and value_targets is not None:
            if config.loss_method == "mse":
                # MSE works best for value function
                values = F.tanh(values)
                loss_value = (values[:, :-1].contiguous().view(-1) - value_targets[:,1:].contiguous().view(-1)) ** 2
                loss_value = loss_value.mean()

            elif config.loss_method == "ce":
                value_reshape = values[:, :-1].contiguous().view(-1, 3)
                value_targets = value_targets[:, 1:].contiguous().view(-1) + 1 # [-1, 0, 1] --> [0, 1, 2]
                loss_value = F.cross_entropy(value_reshape, value_targets.long())

            loss = loss_value

            out = (None, values, (loss, 0, loss_value))
        return out


class ModelConfig(PretrainedConfig):
    activation_function= "relu"
    resid_pdrop = 0.0
    embd_pdrop = 0.0
    attn_pdrop = 0.0
    n_inner = None
    layer_norm_epsilon=1e-5
    initializer_range=0.2
    use_cache = True

    def __init__(self, **kwargs):
        super().__init__(bos_token_id=0, eos_token_id=0)
        self.attrs = ["vocab_size", "n_positions", "n_ctx", "n_embd", "n_layer",
                      "n_head", "activation_function", "resid_pdrop", "embd_pdrop",
                      "attn_pdrop", "layer_norm_epsilon", "n_inner", "initializer_range",
                      "loss_method"]
        for k,v in kwargs.items():
            setattr(self, k, v)
        assert hasattr(self, "loss_method"), "Provide loss method for calculation"

    def __repr__(self):
        return "---- MODEL CONFIGURATION ----\n" + \
            "\n".join([
                f"{k}\t{getattr(self, k)}" for k in list(set(self.attrs))
            ]) + "\n"


class TinyConfig(PretrainedConfig):
    def __init__(self, n_positions, n_ctx, **kwargs):
        self.n_positions = n_positions
        self.n_ctx = n_ctx
        for k,v in kwargs.items():
            setattr(self, k,v)

        self.vocab_size = 2000
        self.n_embd = 18
        self.n_layer = 2
        self.n_head = 2
        self.loss_method = "mse"  # add loss method for values
        self.resid_pdrop = 0.1
        self.embd_pdrop = 0.1
        self.attn_pdrop = 0.1
        self.initializer_range = 0.02


# --- helper functions --- #
def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.2)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, (nn.LayerNorm, Denorm)):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

def configure_optimizers(model, train_config):
    """
    from karpathy/minGPT

    This long function is unfortunately doing something very simple and is being very defensive:
    We are separating out all parameters of the model into two buckets: those that will experience
    weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
    We are then returning the PyTorch optimizer object.
    """
    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear, Conv1D, Attention)
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, Denorm) # add denorm here
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

            if "ValueHead" in fpn: # no decay for value head layers
                no_decay.add(fpn)

            pn_type = pn.split(".")[-1]
            if pn_type == 'bias':
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn_type == 'weight' and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn_type == 'weight' and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay

    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
    assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                % (str(param_dict.keys() - union_params), )

    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=train_config.lr, betas=train_config.betas)
    return optimizer

################################################
####### Trainer ################################
################################################

class Trainer:
    # main training code
    def __init__(self, model, train_dataset, config, test_dataset = None):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config

        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)
            print("Model is now CUDA!")

    def save_checkpoint(self, ckpt_path = None):
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        ckpt_path = ckpt_path if ckpt_path is not None else self.config.ckpt_path
        print(f"Saving Model at {ckpt_path}")
        torch.save(raw_model.state_dict(), ckpt_path)

    def train(self):
        model, config = self.model, self.config
        model_config = self.model.module.config if hasattr(self.model, "module") else self.model.config
        train_data = self.train_dataset
        test_data = self.test_dataset
        num_batches = len(train_data) // config.batch_size + int(len(train_data) % config.batch_size != 0)
        total_steps = num_batches * config.num_epochs

        # initialise wandb here, initially it was init at import and that meant it ran all the time
        wandb.init(project="blindfold-chess")
        wandb.watch(model)

        # create step functions
        model.apply(init_weights)
#         optimizer = torch.optim.Adam(
#             model.parameters(),
#             lr = config.lr,
#             betas = config.betas
#         )
        optimizer = configure_optimizers(model, config) # get AdamW optimiser for this model

        # setup correct scheduler
        if config.scheduler == "CosineAnnealingWarmRestarts":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer = optimizer,
                T_0= total_steps // config.t0div,
                T_mult=config.tmult,
                eta_min=0,
            )
        elif config.scheduler == "OneCycleLR":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer=optimizer,
                max_lr=config.lr,
                total_steps=total_steps
            )

        elif config.scheduler == "MultiStepLR":
            # 102038 at bsize = 35 and maxlen = 170
            # 0      - 0.2     (do not write this in milestones)
            # 1000   - 0.02
            # 10000  - 0.002
            # 50000  - 0.0002
            # 100000 - 0.00002
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer,
                milestones=[1000, 10000, 50000, 100000],
                gamma = 0.1
            )

        elif config.scheduler == "NoamDecay":
            warmup = int(config.warmup_perc * total_steps)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                lr_lambda=lambda e: 100 * (128 ** -0.5) * min(max(1, e) ** -0.5, e * (warmup ** -1.5))
            )
            print("Using NoamDecay scheduler, warmup:", warmup, scheduler)

        elif config.scheduler == "CosineDecay":
            warmup = int(config.warmup_perc * total_steps)
            def lr_lambda(current_step):
                if current_step < warmup:
                    return float(current_step) / float(max(1, warmup))
                progress = float(current_step - warmup) / float(max(1, total_steps - warmup))
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress))) # * 1000
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            print("Using CosineDecay scheduler, warmup:", warmup, scheduler)

        elif config.scheduler == "WarmupConstant":
            warmup = int(config.warmup_perc * total_steps)
            def lr_lambda(current_step: int):
                if current_step < warmup:
                    return float(current_step) / float(max(1.0, warmup))
                return 1.0 # * 10
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            print("Using WarmupConstant scheduler, warmup:", warmup, scheduler)

        elif config.scheduler == "CosineDecayJitter":
            warmup = int(config.warmup_perc * total_steps)
            def lr_lambda(current_step):
                if current_step < warmup:
                    return float(current_step) / float(max(1, warmup))
                progress = float(current_step - warmup) / float(max(1, total_steps - warmup))
                lr = max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress))) # * 1000
                # add jitter in lr
                lr += np.random.uniform(low = -lr/config.jitter_scale, high = lr/config.jitter_scale)
                return lr
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            print("Using CosineDecay scheduler, warmup:", warmup, scheduler)

        elif config.scheduler == "GPT3":
            scheduler = "GPT3"

        else:
            scheduler = None
        print("Train Data Size:", len(train_data), "; Test Data Size:", len(test_data), "; Scheduler:", scheduler)

        # this is second method for creating a training loop
        pbar_train = trange(total_steps, ncols=100)
        dl_train = DataLoader(dataset=train_data, pin_memory=True, batch_size=config.batch_size, shuffle=train_data.shuffle)
        prev_train_loss = 100000 # what was the training loss in previous testing cycle
        no_loss_steps = 0 # no of steps since there was devrease in the loss
        processed_tokens = 0 # number of tokens processed till now
        break_training = False
        train_losses = [-1]
        train_acc = [-1]
        model.train()

        for gs, d in zip(pbar_train, dl_train):
            # total steps is now the primary iteration method
            d = {k:v.to(self.device) for k,v in d.items()}
            pbar_train.set_description(f"[TRAIN] GS: {gs}, Loss: {round(train_losses[-1], 5)}, Acc: {train_acc[-1]}")

            # get model results
            (policy, values, loss) = model(loss=True, **d)
            loss_total = loss[0].mean() # gather
            if not isinstance(loss[1], int):
                loss_policy = loss[1].mean().item() # gather
            else:
                loss_policy = -1
            loss_value = loss[2].mean().item()  # gather
            train_losses.append(loss_total.item())

            # calculate move accuracy
            move_acc = 0
            if policy is not None:
                policy = F.softmax(policy[:,:-1,:], dim = -1).contiguous().view(-1)
                policy = torch.argmax(policy, dim=-1)
                targets = d["input_ids"][:, 1:].contiguous().view(-1)
                move_acc = sum(targets == policy).item()
                move_acc /= targets.size(0)
                train_acc.append(move_acc)

            log_dict = {
                "loss_total": loss_total,
                "loss_policy": loss_policy,
                "loss_value": loss_value,
                "move_acc": move_acc
            }

            # calculate mse error if softmax activation used
            if model_config.loss_method == "ce":
                values = F.softmax(values[:, :-1, :], dim=-1).contiguous().view(-1, 3)  # [B, N, 3]
                values = values.matmul(torch.Tensor([-1, 0, 1]).to(d["value_targets"].device).float())
                mse = (values - d["value_targets"][:,1:].contiguous().view(-1)) ** 2
                log_dict.update({"regression_loss": mse.mean().item()})

            if scheduler is not None and scheduler != "GPT3":
                last_lr = scheduler.get_last_lr()[0]
                log_dict.update({"lr": last_lr})

            # backprop
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            optimizer.step()
            if scheduler is not None and scheduler != "GPT3":
                scheduler.step()

            # ------------- LR SCHEDULING
            elif scheduler == "GPT3":
                # update learning rate
                # number of processed tokens now are the actual number of tokens processed and not just len*batch_size
                processed_tokens += (d["input_ids"] >= 0).sum()
                if processed_tokens < config.warmup_tokens:
                    # linear warmup
                    lr_mult = float(processed_tokens) / float(max(1, config.warmup_tokens))
                else:
                    # cosine learning rate decay
                    progress = float(processed_tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                    lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                lr = config.lr * lr_mult
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                log_dict.update({"lr": lr})
            # ------------- LR SCHEDULING

            # test if time has come
            if gs > 0 and gs % config.test_every == 0:
                model.eval()
                dl_test = DataLoader(
                    dataset = test_data, pin_memory = True, batch_size = config.batch_size, shuffle=test_data.shuffle
                )

                num_batches = len(test_data) // config.batch_size + int(len(test_data) % config.batch_size != 0)

                test_losses = []
                test_acc = []
                test_loss_value = []
                pbar_test = trange(num_batches, ncols = 100)
                for it, d in zip(pbar_test, dl_test):
                    d = {k:v.to(self.device) for k,v in d.items()}
                    pbar_test.set_description(f"[VAL] Global ({gs}) -> [{it + 1}/{num_batches}]")

                    with torch.no_grad():
                        # get model results
                        (policy, values, loss) = model(loss=True, **d)
                        loss_total = loss[0].mean().item()  # gather
                        if not isinstance(loss[1], int):
                            loss_policy = loss[1].mean().item()  # gather
                        else:
                            loss_policy = -1
                        loss_value = loss[2].mean().item()  # gather
                        test_losses.append([loss_total])
                        test_loss_value.append([loss_value])

                        # calculate move accuracy
                        move_acc = 0
                        if policy is not None:
                            policy = F.softmax(policy[:,:-1,:], dim = -1).contiguous().view(-1)
                            policy = torch.argmax(policy, dim=-1)
                            targets = d["input_ids"][:, 1:].contiguous().view(-1)
                            move_acc = sum(targets == policy).item()
                            move_acc /= targets.size(0)
                            test_acc.append(move_acc)

                        if model_config.loss_method == "ce":
                            values = F.softmax(values[:, :-1, :], dim=-1).contiguous().view(-1, 3)  # [B, N, 3]
                            values = values.matmul(torch.Tensor([-1, 0, 1]).to(d["value_targets"].device).float())
                            mse = (values - d["value_targets"][:,1:].contiguous().view(-1)) ** 2
                            test_losses[-1].append(mse.mean().item())

                # now testing is complete so see the results, save or stop if needed
                test_losses = np.array(test_losses)
                losses = np.mean(test_losses[:, 0])
                test_acc = np.mean(test_acc)
                print(f"Loss: {losses:.3f}; Acc: {test_acc:.3f}", end = " ")
                log_dict.update({
                    "test_loss": losses,
                    "test_acc": test_acc
                })
                if model_config.loss_method == "ce":
                    log_dict.update({"test_regression_loss": np.mean(test_losses[:, 1])})
                else:
                    log_dict.update({"test_value_loss": np.mean(test_loss_value)})

                if prev_train_loss > losses:
                    prev_train_loss = losses
                    no_loss_steps = 0
                    print("... previous loss was larger, updating values")
                    cp = config.ckpt_path.replace(".pt", f"_{gs}.pt")
                    self.save_checkpoint(cp)
                else:
                    no_loss_steps += 1
                    print(f"... previous loss was smaller. No improvements for past {no_loss_steps} evaluations")

                if config.patience != -1 and  no_loss_steps == config.patience:
                    break_training = True

                model.train()

            wandb.log(log_dict)
            if break_training:
                print("Stopping training")
                break

        model.train()
        print("Final Step Save")
        cp = config.ckpt_path.replace(".pt", f"_train_final.pt")
        self.save_checkpoint(cp)

class TrainerConfig:
    num_epochs = 2
    batch_size = 64
    lr = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    num_workers = 0 # for DataLoader
    ckpt_path = None
    tb_path = None
    patience = -1
    test_every = None
    scheduler = None
    weight_decay = 0.1
    warmup_perc = None
    warmup_tokens = None
    final_tokens = None

    def __init__(self, **kwargs):
        self.attrs = [
            "num_epochs", "batch_size", "lr", "betas", "grad_norm_clip", "num_workers",
            "ckpt_path", "tb_path", "patience", "test_every", "scheduler", "weight_decay",
            "warmup_perc", "final_tokens", "warmup_tokens"
        ]
        for k,v in kwargs.items():
            setattr(self, k, v)
            self.attrs.append(k)

        if self.scheduler == "CosineAnnealingWarmRestarts":
            assert hasattr(self, "t0div"), "Provide this if using CosineAnnealingWarmRestarts Scheduler"
            assert hasattr(self, "tmult"), "Provide this if using CosineAnnealingWarmRestarts Scheduler"

        elif self.scheduler in ["NoamDecay", "CosineDecay", "WarmupConstant"]:
            assert hasattr(self, "warmup_perc"), "Provide Warmup percentage"

        elif self.scheduler in ["CosineDecayJitter"]:
            assert hasattr(self, "warmup_perc"), "Provide Warmup percentage"
            assert hasattr(self, "jitter_scale"), "Provide jitter scale"

        if self.warmup_tokens == None:
            # total tokens // (batch_size * 170)
            # this is the final tokens you get when you train full game dataset on batch size 40
            # and 170 tokens in each sequence
            self.final_tokens = 185132*40*170  # total size of all the tokens
            
            # this is the final tokens you get when you train on GPT style block
            # self.final_tokens = 613256130  # total size of all the tokens

            self.warmup_tokens = int(self.final_tokens * self.warmup_perc)
            print("Auto Setting warmup_tokens using", self.warmup_perc, "to", self.warmup_tokens)

        elif self.scheduler == "GPT3":
            assert self.final_tokens != None
            assert self.warmup_tokens != None

    def __repr__(self):
        return "---- TRAINER CONFIGURATION ----\n" + \
            "\n".join([f"{k}\t{getattr(self, k)}" for k in list(set(self.attrs))
        ]) + "\n"


################################################
####### Dataset ################################
################################################
class FullDatasetPreLoaded(Dataset):
    def __init__(self, lms, results, m2id, pos = None):
        self.lms = lms
        self.results = results
        self.m2id = m2id
        self.pos = pos
        self.shuffle = True

    def __len__(self):
        return self.lms.shape[0]

    def __getitem__(self, index):
        ret_dict = {
            "input_ids": torch.from_numpy(self.lms[index]).long(),
            "value_targets": torch.from_numpy(self.results[index]).float()
        }
        if self.pos != None:
            ret_dict.update({
                "position_ids": torch.from_numpy(self.pos[index]).long(),
            })
        return ret_dict


def get_datasets(config, split):
    """This function returns two datasets one for training and another for holdout
    No need to load to different datasets or create a split between them"""
    with open(config.m2id, "r") as m:
        m2id = json.load(m)
        if "[GAME]" not in m2id:  # only if not found
            GAME = len(m2id)
            m2id["[GAME]"] = GAME  # new game flag
        else:
            GAME = m2id["[GAME]"]

    # init position IDs with None
    pos = None
    if config.lm[-4:] == "hdf5":
        # this is the hdf5
        st = time.time()
        data = h5py.File(config.lm, "r")
        lms = data["lms"]
        results = data["res"]
        print(f"HDF5 Loading took: {time.time()-st}s")

    elif config.lm[-3:] == "npz":
        # this is numpy zip, load the pickle
        st = time.time()
        clm = np.load("data/clm.npz")
        lms = clm["lms"]
        results = clm["res"]
        try:
            pos = clm["pos"]
        except:
            pos = None

        lms = lms.reshape(-1, config.maxlen)
        results = results.reshape(-1, config.maxlen)

        print(f"Numpy Loading took: {time.time()-st}s")

    else:
        len_file = 0
        with open(config.lm, "r") as f:
            for _ in f:
                len_file += 1
        total_len = len_file

        # now load the complete dataset in memory
        with open(config.lm, "r") as flm, open(config.rf, "r") as fres:
            lms = [] # all the sequences
            results = [] # all the results
            print("Loading samples in memory ... this will take some time")
            for idx, lm, game_res in zip(trange(total_len), flm, fres):
                # ignore BOS + EOS tags, [GAME] does it for us
                lm = list(map(lambda x: int(x.strip()), lm.split()))[1:-1]
                lms.extend([GAME] + lm)

                # get the targets for values as [0,res,-res,res,-res...]
                game_res = float(game_res)
                res = np.ones(len(lm)) * game_res
                res[np.arange(1, len(lm), 2)] = -game_res
                results.extend([0] + res.tolist()) # first will always generate 0

                # used for batch size testing on GPUs, full loading takes tine
                # if len(results) > 90 * 10000:
                #    break

        # now convert this long list to sequence
        lms = np.array(lms[:-(len(lms) % config.maxlen)]).reshape(-1, config.maxlen)
        results = np.array(results[:-(len(results) % config.maxlen)]).reshape(-1, config.maxlen)

    print(f"Moves: {lms.shape}; Results: {results.shape}")

    test_idx = int(split * lms.shape[0])
    ds_train = FullDatasetPreLoaded(lms[test_idx:], results[test_idx:], m2id, pos)
    ds_test = FullDatasetPreLoaded(lms[:test_idx], results[:test_idx], m2id, pos)
    return ds_train, ds_test

#
# This method below uses the complete sequence of the game from start to end
# hope is that with this better representation the game play would be better
#
class FullGameLoadedDataset(Dataset):
    def __init__(self, config, lms, results, m2id):
        self.config = config
        self.lms = lms
        self.res = results
        self.m2id = m2id
        self.id2m = {v:k for k,v in self.m2id.items()}
        self.mask_size = len(self.m2id)
        self.shuffle = True

        with open(config.m2id, "r") as m:
            m2id = json.load(m)
            if "[GAME]" not in m2id:  # only if not found
                GAME = len(m2id)
                m2id["[GAME]"] = GAME  # new game flag
            else:
                GAME = m2id["[GAME]"]

            if "[END_GAME]" not in m2id:  # only if not found
                END_GAME = len(m2id)
                m2id["[END_GAME]"] = END_GAME  # new game flag
            else:
                END_GAME = m2id["[END_GAME]"]

        # add the game tag and the end game tag
        self.GAME = GAME
        self.END_GAME = END_GAME

        # print("GAME",self.GAME)
        # print("END_GAME",self.END_GAME)

        self.board = chess.Board()

    def __len__(self):
        return len(self.lms) #.shape[0]

    def get_move_mask(self, m, l):
        self.board.reset()
        for i,mv in enumerate(l[l>0]):
            m[i][[self.m2id[str(x)[:4]] for x in self.board.legal_moves]] = 0
            mid = self.id2m[mv.item()]
            if mid == "[END_GAME]":
                break
            self.board.push(Move(mid))
        return m

    def __getitem__(self, i):
        self.board.reset() # inplace reset
        config = self.config
        m = self.config.maxlen
        G = self.GAME
        EG = self.END_GAME
        lm = self.lms[i]
        if isinstance(lm, np.ndarray):
            lm = lm.tolist()
        if isinstance(lm, list):
            lm = [G] + lm
        res = self.res[i]

        # lm --> [G] + [m1, m2 ... mn] + [EG]
        lm = lm + [EG]
        upto_id = len(lm)

        # get the targets for values as [res,-res,res,-res...]
        if isinstance(res, (float, int, str)):
            game_res = float(res)
            res = np.ones(len(lm)) * game_res
            res[2:len(lm):2] = -game_res
            res[[0, -1]] = 0 # first and last tag [GAME] has to predict 0
            res = res.tolist()
        else:
            res = res.tolist()

        # create the attention mask
        am = [1 for _ in range(len(lm)-1)] # attention mask

        # now handle smaller longer sequences
        if len(lm) > m + 1:
            lm = lm[:m+1]
            res = res[:m]
            am = am[:m] # m+1-1 = m

        # need to perform padding
        elif len(lm) < m + 1:
            to_add = m+1-len(lm)
            # print(m, to_add, len([0 for _ in range(to_add)]), res, lm)
            lm = lm + [G for _ in range(to_add)]
            res = res + [0 for _ in range(to_add)]
            am  = am + [0 for _ in range(to_add)]

        am = torch.Tensor(am).long()[:m]
        val_target = torch.Tensor(res).float()[:m]
        labels = torch.Tensor(lm[1:]).long()
        if upto_id < m:
            labels[upto_id-1:] = -100
        lm = torch.Tensor(lm[:-1]).long()[:m]

        # if config.ret_legal_mask: # always return
        # we do not want the model to learn all combination over all the possible moves
        # just the moves that are legal. So we perform masking, this is done on the lines
        # of AlphaZero which does recieve move mask as a part of the board state
        labels_mask = torch.ones(len(lm), self.mask_size) * -1e8
        labels_mask = self.get_move_mask(labels_mask, labels)

        assert len(labels) == m, f"len-->{labels.size()}, {i}"
        assert len(lm) == m, f"len-->{lm.size()}, {i}"
        assert len(am) == m, f"len-->{am.size()}, {i}"
        assert len(val_target) == m, f"len-->{val_target.size()}, {i}"
        assert len(labels_mask) == m, f"len-->{labels_mask.size()}, {i}"

        return {
            "input_ids": lm,
            "attention_mask": am,
            "value_targets": val_target,
            "labels": labels,
            "labels_mask": labels_mask
        }

def get_datasets_full_game(config, split):
    # same as get_datasets but for FullGame
    with open(config.m2id, "r") as m:
        m2id = json.load(m)
        if "[GAME]" not in m2id:  # only if not found
            GAME = len(m2id)
            m2id["[GAME]"] = GAME  # new game flag
        else:
            GAME = m2id["[GAME]"]

    st = time.time()
    print(":: Starting Loading")

    if config.lm_train[-3:] == "npz":
        clm = np.load("data/clm.npz")
        lms = clm["lms"]
        results = clm["res"]
        split_idx = np.argwhere(lms.flatten() == 0.0).flatten()[1:]
        lms_split = np.split(lms.flatten(), split_idx)
        res_split = np.split(results.flatten(), split_idx)
        print(f"Numpy Loading took: {time.time()-st}s")

    if config.lm_train[-2:] == ".p":
        with open("data/final.p", "r") as p:
            data = pickle.load(p)
            lms_split = data["lms"]
            res_split = data["res"]
        print(f"Pickle Loading took: {time.time()-st}s")

    test_idx = int(split * lms_split.shape[0])
    ds_train = FullGameLoadedDataset(config, lms_split[test_idx:], res_split[test_idx:], m2id)
    ds_test = FullGameLoadedDataset(config, lms_split[:test_idx], res_split[:test_idx], m2id)
    return ds_train, ds_test


class ChessData(IterableDataset):
    def __init__(self, config):
        len_file = 0
        with open(config.lm, "r") as f:
            for _ in f:
                len_file += 1
        self.len = len_file

        with open(config.m2id, "r") as m:
            self.m2id = json.load(m)
            if "[GAME]" not in self.m2id: # only if not found
                self.GAME = len(self.m2id)
                self.m2id["[GAME]"] = self.GAME # new game flag
            else:
                self.GAME = self.m2id["[GAME]"]

        self.id2m = {i:m for i,m in self.m2id.items()}
        self.config = config
        self.shuffle = False

    def __len__(self):
        return self.len

    def _update_m2id(self, key, value):
        if key not in self.m2id:
            self.m2id.update({key: value})

    @staticmethod
    def _sliding_buckets(x, s):
        # return buckets of size seqlen
        return [x[i*s:(i+1)*s] for i in range((len(x) // s) + min(len(x) % s, 1))]

    def __iter__(self):
        config = self.config
        with open(config.lm, "r") as flm, open(config.rf, "r") as fres:
            lms = [] # all the sequences
            results = [] # all the results
            for lm, game_res in zip(flm, fres):
                # ignore BOS + EOS tags, [GAME] does it for us
                lm = list(map(lambda x: int(x.strip()), lm.split()))[1:-1]
                lms.extend([self.GAME] + lm)

                # get the targets for values as [0,res,-res,res,-res...]
                game_res = float(game_res)
                res = np.ones(len(lm)) * game_res
                res[np.arange(1, len(lm), 2)] = -game_res
                results.extend([0] + res.tolist()) # first will always generate 0

                if len(lms) > config.buffer:
                    # print("\n\n\n", len(lms), len(results))
                    # print(len(lm), len(res))

                    # no of samples
                    batches = len(lms) // config.maxlen
                    samples = self._sliding_buckets(
                        np.asarray(lms[:config.maxlen * batches]),
                        config.maxlen
                    )
                    values = self._sliding_buckets(
                        np.asarray(results[:config.maxlen * batches]),
                        config.maxlen
                    )
                    idx = np.arange(len(values))
                    np.random.shuffle(idx)
                    for i in idx:
                        out = {
                            "input_ids": torch.from_numpy(samples[i]).long(),
                            "value_targets": torch.from_numpy(values[i]).float()
                        }
                        yield out
                    del lms[:config.maxlen * batches]
                    del results[:config.maxlen * batches]


class ChessDataInMemory(Dataset):
    def __init__(self, config):
        len_file = 0
        with open(config.lm, "r") as f:
            for _ in f:
                len_file += 1
        self.len = len_file

        with open(config.m2id, "r") as m:
            self.m2id = json.load(m)
            if "[GAME]" not in self.m2id:  # only if not found
                self.GAME = len(self.m2id)
                self.m2id["[GAME]"] = self.GAME  # new game flag
            else:
                self.GAME = self.m2id["[GAME]"]

        self.id2m = {i: m for i,m in self.m2id.items()}
        self.config = config

        # now load the complete dataset in memory
        with open(config.lm, "r") as flm, open(config.rf, "r") as fres:
            lms = [] # all the sequences
            results = [] # all the results
            print("Loading samples in memory ... this will take some time")
            for idx, lm, game_res in zip(trange(self.len), flm, fres):
                # ignore BOS + EOS tags, [GAME] does it for us
                lm = list(map(lambda x: int(x.strip()), lm.split()))[1:-1]
                lms.extend([self.GAME] + lm)

                # get the targets for values as [0,res,-res,res,-res...]
                game_res = float(game_res)
                res = np.ones(len(lm)) * game_res
                res[np.arange(1, len(lm), 2)] = -game_res
                results.extend([0] + res.tolist()) # first will always generate 0

        # now convert this long list to sequences
        # using ChessData._sliding_buckets() is just ridiculously slow will have to use np.reshape
        # method, but for this will need to drop the last few tokens.
        # self.lms = np.asarray(ChessData._sliding_buckets(lms, config.maxlen))
        # self.results = np.asarray(ChessData._sliding_buckets(results, config.maxlen))

        self.lms = np.array(lms[:-(len(lms) % config.maxlen)]).reshape(-1, config.maxlen)
        self.results = np.array(results[:-(len(results) % config.maxlen)]).reshape(-1, config.maxlen)

        print(f"Moves: {self.lms.shape}; Results: {self.results.shape}")

        assert self.lms.shape == self.results.shape
        self.shuffle = True

    def __len__(self):
        return self.lms.shape[0]

    def __getitem__(self, index):
        return {
            "input_ids": torch.from_numpy(self.lms[index]).long(),
            "value_targets": torch.from_numpy(self.results[index]).float()
        }

class DataConfig:
    lm = None
    rf = None # results file
    m2id = None
    maxlen = None
    buffer= None
    ret_legal_mask=True

    def __init__(self, **kwargs):
        self.attrs = ["lm", "rf", "m2id", "maxlen", "buffer"]
        for k,v in kwargs.items():
            setattr(self, k, v)
            self.attrs.append(k)

    def __repr__(self):
        return "---- DATA CONFIGURATION ----\n" + \
            "\n".join([f"{k}\t{getattr(self, k)}" for k in list(set(self.attrs))
        ]) + "\n"

