"""chess lm model + dataset file"""

import h5py
import json
import time
import random
import numpy as np
from tqdm import trange

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import IterableDataset, DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from transformers import GPT2Model, GPT2Config as ModelConfig
from transformers.modeling_gpt2 import MLP, Attention

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
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gpt = GPT2Model(config)
        self.policy_head = nn.Linear(config.n_embd, config.vocab_size)
        # self.value_head = nn.Linear(config.n_embd, 1)
        self.value_head = nn.Sequential(*[
            nn.Linear(config.n_embd, config.n_embd // 2),
            nn.LayerNorm(config.n_embd // 2, eps=config.layer_norm_epsilon),
            nn.ReLU(),
            nn.Linear(config.n_embd // 2, 1),
            nn.Tanh()
        ])

    def forward(self, input_ids, value_targets = None, loss = None, **gptkwargs):
        x = self.gpt(input_ids, return_dict = True, **gptkwargs)
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


class PolicyHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = Attention(config.n_embd, config.n_ctx, config, scale = True)
        self.ln = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = MLP(config.n_embd * 4, config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        
    def forward(self, hidden_states):
        # this is the Block
        attn_output = self.attn(hidden_states)[0]
        hidden_states = attn_output + hidden_states # residual connection
        feed_forward_hidden_states = self.mlp(self.ln(hidden_states))
        hidden_states = hidden_states + feed_forward_hidden_states # residual connection
        return self.lm_head(hidden_states)
        

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
        train_data = self.train_dataset
        test_data = self.test_dataset
        num_batches = len(train_data) // config.batch_size + int(len(train_data) % config.batch_size != 0)
        total_steps = num_batches * config.num_epochs
        
        # create step functions
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr = config.lr,
            betas = config.betas
        )
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps = total_steps, max_lr=0.05)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
            milestones=[1000,2000,6000,10000]
        , gamma=0.1) # 26455
        
        print("Train Data Size:", len(train_data), "; Test Data Size:", len(test_data))

        with SummaryWriter(log_dir=config.tb_path, flush_secs=20) as tb:
            # this is second method for creating a training loop
            pbar_train = trange(total_steps, ncols=100)
            dl_train = DataLoader(dataset=train_data, pin_memory=True, batch_size=config.batch_size, shuffle=train_data.shuffle)
            prev_train_loss = 100000 # what was the training loss in previos testing cycle
            no_loss_steps = 0 # no of steps since there was devrease in the loss
            break_training = False
            train_losses = [-1]
            train_acc = [-1]
            model.train()
            
            for gs, d in zip(pbar_train, dl_train):
                # total steps is now the primary iteration method
                d = {k:v.to(self.device) for k,v in d.items()}
                
                epoch = gs // config.batch_size
                pbar_train.set_description(f"[TRAIN] GS: {gs}, Epoch: {epoch}, Loss: {round(train_losses[-1], 5)}, Acc: {train_acc[-1]}")

                # get model results
                (policy, values, loss) = model(loss=True, **d)
                loss_total = loss[0].mean() # gather
                loss_policy = loss[1].mean() # gather
                loss_value = loss[2].mean() # gather
                train_losses.append(loss_total.item())

                # calculate move accuracy
                policy = F.softmax(policy[:,:-1,:], dim = -1).contiguous().view(-1)
                policy = torch.argmax(policy, dim=-1)
                targets = d["input_ids"][:, 1:].contiguous().view(-1)
                move_acc = sum(targets == policy).item()
                move_acc /= targets.size(0)
                train_acc.append(move_acc)

                # add to tensorboard
                tb.add_scalar("train/loss_total", loss_total.item(), global_step=gs, walltime=time.time())
                tb.add_scalar("train/loss_policy", loss_policy.item(), global_step=gs, walltime=time.time())
                tb.add_scalar("train/loss_value", loss_value.item(), global_step=gs, walltime=time.time())
                tb.add_scalar("train/move_acc", move_acc, global_step=gs, walltime=time.time())
                tb.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step=gs, walltime=time.time())

                # backprop
                loss_total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                optimizer.step()
                scheduler.step()

                # test if time has come
                if gs > 0 and gs % config.test_every == 0:
                    model.eval()
                    dl_test = DataLoader(
                        dataset = test_data, pin_memory = True, batch_size = config.batch_size, shuffle=test_data.shuffle
                    )

                    num_batches = len(test_data) // config.batch_size + int(len(test_data) % config.batch_size != 0)

                    test_losses = []
                    test_acc = []
                    pbar_test = trange(num_batches, ncols = 100)
                    for it, d in zip(pbar_test, dl_test):
                        d = {k:v.to(self.device) for k,v in d.items()}
                        pbar_test.set_description(f"[VAL] Global ({gs}) -> [{it + 1}/{num_batches}]")

                        with torch.no_grad():
                            # get model results
                            (policy, values, loss) = model(loss=True, **d)
                            loss_total = loss[0].mean() # gather
                            loss_policy = loss[1].mean() # gather
                            loss_value = loss[2].mean() # gather
                            test_losses.append(loss_total.item())

                            # calculate move accuracy
                            policy = F.softmax(policy[:,:-1,:], dim = -1).contiguous().view(-1)
                            policy = torch.argmax(policy, dim=-1)
                            targets = d["input_ids"][:, 1:].contiguous().view(-1)
                            move_acc = sum(targets == policy).item()
                            move_acc /= targets.size(0)
                            test_acc.append(move_acc)

                            # add to tensorboard
                            tb.add_scalar("test/loss_total", loss_total.item(), global_step=gs, walltime=time.time())
                            tb.add_scalar("test/loss_policy", loss_policy.item(), global_step=gs, walltime=time.time())
                            tb.add_scalar("test/loss_value", loss_value.item(), global_step=gs, walltime=time.time())
                            tb.add_scalar("test/move_acc", move_acc, global_step=gs, walltime=time.time())

                    # now testing is complete so see the results, save or stop if needed
                    losses = np.mean(test_losses)
                    test_acc = np.mean(test_acc)
                    print(f"Loss: {losses:.3f}; Acc: {test_acc:.3f}", end = " ")

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

                if break_training:
                    print("Stopping training")
                    break


            ### The code below is saved for reference I don't use this method anymore

            # def run_epoch(split, _gs = None):
            #     is_train = split == "train"
            #     model.train(is_train)
            #     data = self.train_dataset if is_train else self.test_dataset
            #     dl = DataLoader(
            #         data,
            #         pin_memory = True,
            #         batch_size = config.batch_size,
            #         shuffle = data.shuffle
            #     )

            #     num_batches = len(data) // config.batch_size + int(len(data) % config.batch_size != 0)

            #     losses = []
            #     pbar = trange(num_batches, ncols = 100)
            #     for it, d in zip(pbar, dl):
            #         _l = -1 if not losses else losses[-1]
            #         if is_train:
            #             pbar.set_description(f"[TRAIN] GS: {_gs}, IT: {it}, Loss: {round(_l, 5)}")
            #         else:
            #             pbar.set_description(f"[VAL] Epoch: {_gs}")

            #         d = {k:v.to(self.device) for k,v in d.items()}
            #         # print({k:v.size() for k,v in d.items()})
            #         with torch.set_grad_enabled(is_train):
            #             (policy, values, loss) = model(loss=True, **d)
            #             loss_total = loss[0].mean() # gather
            #             loss_policy = loss[1].mean() # gather
            #             loss_value = loss[2].mean() # gather
            #             losses.append(loss_total.item())

            #             policy = F.softmax(policy[:,:-1,:], dim = -1).contiguous().view(-1)
            #             values = values[:,:-1,0].contiguous().view(-1)
            #             move_acc = sum(d["input_ids"][:, 1:].contiguous().view(-1), torch.argmax(policy, dim=-1)).item()

            #         # save if required
            #         if _gs % config.save_every == 0:
            #             cp = config.ckpt_path.replace(".pt", f"_{_gs}.pt")
            #             self.save_checkpoint(cp)

            #         if is_train:
            #             # add things to tb, loss and attention images
            #             tb.add_scalar("train/loss_total", loss_total.item(), global_step=_gs, walltime=time.time())
            #             tb.add_scalar("train/loss_policy", loss_policy.item(), global_step=_gs, walltime=time.time())
            #             tb.add_scalar("train/loss_value", loss_value.item(), global_step=_gs, walltime=time.time())
            #             tb.add_scalar("train/move_acc", move_acc, global_step=_gs, walltime=time.time())

            #             loss_total.backward()
            #             torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            #             optimizer.step()
            #             _gs += 1

            #         else:
            #             # add to tensorboard
            #             tb.add_scalar("test/loss_total", loss_total.item(), global_step=_gs, walltime=time.time())
            #             tb.add_scalar("test/loss_policy", loss_policy.item(), global_step=_gs, walltime=time.time())
            #             tb.add_scalar("test/loss_value", loss_value.item(), global_step=_gs, walltime=time.time())
            #             tb.add_scalar("test/move_acc", move_acc, global_step=_gs, walltime=time.time())

            #     if not is_train:
            #         test_loss = float(np.mean(losses))
            #         return test_loss

            #     return _gs

            # # now write wrapper for each epoch
            # gs = 0
            # for e in range(config.max_epochs):
            #     gs = run_epoch("train", gs)
            #     self.save_checkpoint()

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

    def __init__(self, **kwargs):
        self.attrs = [
            "num_epochs",
            "batch_size",
            "learning_rate",
            "betas",
            "grad_norm_clip",
            "num_workers",
            "ckpt_path",
            "tb_path",
            "patience",
            "test_every"
        ]
        for k,v in kwargs.items():
            setattr(self, k, v)
            self.attrs.append(k)

    def __repr__(self):
        return "---- TRAINER CONFIGURATION ----\n" + \
            "\n".join([f"{k}\t{getattr(self, k)}" for k in list(set(self.attrs))
        ]) + "\n"


################################################
####### Dataset ################################
################################################
class FullDatasetPreLoaded(Dataset):
    def __init__(self, lms, results, m2id):
        self.lms = lms
        self.results = results
        self.m2id = m2id
        self.shuffle = True

    def __len__(self):
        return self.lms.shape[0]

    def __getitem__(self, index):
        return {
            "input_ids": torch.from_numpy(self.lms[index]).long(),
            "value_targets": torch.from_numpy(self.results[index]).float()
        }

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
    ds_train = FullDatasetPreLoaded(lms[test_idx:], results[test_idx:], m2id)
    ds_test = FullDatasetPreLoaded(lms[:test_idx], results[:test_idx], m2id)
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

    def __init__(self, **kwargs):
        self.attrs = []
        for k,v in kwargs.items():
            setattr(self, k, v)
            self.attrs.append(k)

    def __repr__(self):
        return "---- DATA CONFIGURATION ----\n" + \
            "\n".join([f"{k}\t{getattr(self, k)}" for k in list(set([
                "lm", "rm", "m2id", "maxlen", "buffer"
            ] + self.attrs))
        ]) + "\n"

