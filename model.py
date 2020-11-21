"""chess lm model + dataset file"""

import json
import time
import random
import numpy as np
from tqdm import trange

import torch
from torch import nn
from torch.optim import Adam
from torch.nn import functional as F
from torch.utils.data import IterableDataset, DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import GPT2Model, GPT2Config as ModelConfig

################################################
####### Model ##################################
################################################


class BaseHFGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gpt = GPT2Model(config)
        self.policy_head = nn.Linear(config.n_embd, config.vocab_size, bias = False)
#         self.value_head = nn.Sequential(
#             nn.Linear(config.n_embd, config.n_embd // 2),
#             nn.ReLU(),
#             nn.Linear(config.n_embd // 2, 1),
#         )
        self.value_head = nn.Linear(config.n_embd, 1)

    def forward(self, input_ids, value_targets = None, loss = None, **gptkwargs):
        x = self.gpt(input_ids, return_dict = True, **gptkwargs)
        logits = self.policy_head(x.last_hidden_state)
        values = torch.tanh(self.value_head(x.last_hidden_state))
        out = (logits, x)
        if loss is not None and value_targets is not None:            
            # Categorical cross entropy loss worked best for policy
            logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
            targets = input_ids[:, 1:].contiguous().view(-1)
            loss_policy = F.cross_entropy(logits, targets)
            
            # MSE works best for value function
            loss_value = (values[:, :-1].contiguous().view(-1) - value_targets[:,1:].contiguous().view(-1)) ** 2
            loss_value = loss_value.mean()

            loss = loss_policy + loss_value
            
            out = (logits, x, (loss, loss_policy, loss_value))
        return out

################################################
####### Trainer ################################
################################################

class Trainer:
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
        print(f"Saving Model at {ckpt_path}")
        ckpt_path = ckpt_path if ckpt_path is not None else self.config.ckpt_path
        torch.save(raw_model.state_dict(), ckpt_path)

    def train(self):
        model, config = self.model, self.config
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr = config.learning_rate,
            betas = config.betas
        )
        
        with SummaryWriter(log_dir=config.tb_path, flush_secs=20) as tb:
            def run_epoch(split, _gs = None):
                is_train = split == "train"
                model.train(is_train)
                data = self.train_dataset if is_train else self.test_dataset
                dl = DataLoader(
                    data,
                    pin_memory = True,
                    batch_size = config.batch_size,
                    shuffle = data.shuffle
                )
                
                num_batches = len(data) // config.batch_size + int(len(data) % config.batch_size != 0)

                losses = []
                pbar = trange(num_batches, ncols = 100)
                for it, d in zip(pbar, dl):
                    _l = -1 if not losses else losses[-1]
                    if is_train:
                        pbar.set_description(f"[TRAIN] GS: {_gs}, IT: {it}, Loss: {round(_l, 5)}")
                    else:
                        pbar.set_description(f"[VAL] Epoch: {_gs}")
                        
                    d = {k:v.to(self.device) for k,v in d.items()}
                    
                    # print({k:v.size() for k,v in d.items()})

                    with torch.set_grad_enabled(is_train):
                        (out, _, loss) = model(loss = True, **d)
                        loss_total = loss[0].mean() # gather
                        loss_policy = loss[1].mean() # gather
                        loss_value = loss[2].mean() # gather
                        losses.append(loss_total.item())
                        
                    # save if required
                    if _gs % config.save_every == 0:
                        cp = config.ckpt_path.replace(".pt", f"_{_gs}.pt")
                        self.save_checkpoint(cp)
                    

                    if is_train:
                        # add things to tb, loss and attention images
                        tb.add_scalar("loss/loss_total", loss_total.item(), global_step=_gs, walltime=time.time())
                        tb.add_scalar("loss/policy", loss_policy.item(), global_step=_gs, walltime=time.time())
                        tb.add_scalar("loss/value", loss_value.item(), global_step=_gs, walltime=time.time())

                        loss_total.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                        optimizer.step()
                        _gs += 1
                        
                
                if not is_train:
                    test_loss = float(np.mean(losses))
                    return test_loss

                return _gs

            # now write wrapper for each epoch
            gs = 0
            for e in range(config.max_epochs):
                gs = run_epoch("train", gs)
                self.save_checkpoint()

class TrainerConfig:
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    num_workers = 0 # for DataLoader
    ckpt_path = None
    tb_path = None

    def __init__(self, **kwargs):
        self.attrs = []
        for k,v in kwargs.items():
            setattr(self, k, v)
            self.attrs.append(k)

    def __repr__(self):
        return "---- TRAINER CONFIGURATION ----\n" + \
            "\n".join([f"{k}\t{getattr(self, k)}" for k in list(set([
                "max_epochs",
                "batch_size",
                "learning_rate",
                "betas",
                "grad_norm_clip",
                "num_workers",
            ] + self.attrs))
        ]) + "\n"

def set_seed(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


################################################
####### Dataset ################################
################################################


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

    def __init__(self, **kwargs):
        self.attrs = []
        for k,v in kwargs.items():
            setattr(self, k, v)
            self.attrs.append(k)

    def __repr__(self):
        return "---- TRAINER CONFIGURATION ----\n" + \
            "\n".join([f"{k}\t{getattr(self, k)}" for k in list(set([
                "lm", "rm", "m2id", "maxlen", "buffer"
            ] + self.attrs))
        ]) + "\n"

