"""chess lm model + dataset file"""

import json
import time
import random
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch.optim import Adam
from torch.nn import functional as F
from torch.utils.data import IterableDataset, DataLoader
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
        self.lmhead = nn.Linear(config.n_embd, config.vocab_size, bias = False)

    def forward(self, input_ids, loss = None, **gptkwargs):
        x = self.gpt(input_ids, return_dict = True, **gptkwargs)
        logits = self.lmhead(x.last_hidden_state)
        out = (logits, x)
        if loss is not None:
            logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
            targets = input_ids[:, 1:].contiguous().view(-1)
            loss = F.cross_entropy(logits, targets)
            out = (logits, x, loss)
        return out



################################################
####### Trainer ################################
################################################

class Trainer:
    def __init__(self, model, train_dataset, test_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config

        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def save_checkpoint(self):
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        print(f"Saving Model at {self.config.ckpt_path}")
        torch.save(raw_model.state_dict(), self.config.ckpt_path)

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
                    batch_size = config.batch_size
                )

                losses = []
                pbar = tqdm(enumerate(dl))
                for it, d in pbar:
                    _l = -1 if not losses else losses[-1]
                    if is_train:
                        pbar.set_description(f"[TRAIN] GS: {_gs}, IT: {it}, Loss: {round(_l, 5)}")
                    else:
                        pbar.set_description(f"[VAL] Epoch: {_gs}")

                    with torch.set_grad_enabled(is_train):
                        (out, _, loss) = model(loss = True, **d)
                        losses.append(loss.item())

                    if is_train:
                        # add things to tb, loss and attention images
                        tb.add_scalar("loss", loss.item(), global_step=_gs, walltime=time.time())
                        # for l, att in enumerate(attentions):
                        #     tb.add_image(
                        #         f"attention/layer_{l}", att[0][0],
                        #         global_step=gs, walltime=time.time(),
                        #         dataformats= "HW"
                        #     )

                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                        optimizer.step()
                        _gs += 1
                
                if not is_train:
                    test_loss = float(np.mean(losses))
                    return test_loss

                return _gs

            # now write wrapper for each epoch
            best_loss = float("inf")
            gs = 1
            for e in range(config.max_epochs):
                gs = run_epoch("train", gs)
                if self.test_dataset is not None:
                    test_loss = run_epoch("test", e)
                    print(f"Test loss: {test_loss}")
                
                # early stopping based on the test loss of just save always if no test set is provided
                good_model = self.test_dataset is None or test_loss < best_loss
                if self.config.ckpt_path is not None and good_model:
                    best_loss = test_loss
                    self.save_checkpoint()


class TrainerConfig:
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    num_workers = 0 # for DataLoader

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
        self.id2m = {i:m for i,m in self.m2id.items()}
        self.config = config

    def __len__(self):
        return self.len

    def _update_m2id(self, key, value):
        if key not in self.m2id:
            self.m2id.update({key: value})

    @staticmethod
    def _sliding_buckets(x, s):
        # return buckets of size seqlen
        return [x[i*s:(i+1)*s] for i in range((len(x) // s) + min(len(x) % s, 1))]

    # @staticmethod
    # def _rolling_window(a, window_size):
    #     shape = a.shape[:-1] + (a.shape[-1] - window_size + 1, window_size)
    #     strides = a.strides + (a.strides[-1],)
    #     return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

    def __iter__(self):
        config = self.config
        with open(config.lm, "r") as flm:
            lms = []
            for lm in flm:    
                lm = list(map( lambda x: int(x.strip()), lm.split() ))
                lms.extend(lm)
                if len(lms) > config.buffer:
                    # no of samples
                    batches = len(lms) // config.maxlen
                    samples = self._sliding_buckets(
                        np.asarray(lms[:config.maxlen * batches]),
                        config.maxlen
                    )
                    np.random.shuffle(samples)
                    for s in samples:
                        out = {"input_ids": torch.from_numpy(s).long()}
                        yield out
                    del lms[:config.maxlen * batches]

class DataConfig:
    lm = None
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
                "lm",
                "m2id",
                "maxlen",
                "buffer",
            ] + self.attrs))
        ]) + "\n"




# def accuracy(b, logits):
#     # (upto -1) compared to (from 1)
#     if CUDA and APEX:
#         input_ids = b["input_ids"].detach().cpu().numpy()
#         pred_ids = torch.argmax(logits, dim=-1).detach().cpu().numpy()
#     else:
#         input_ids = b["input_ids"].detach().numpy()
#         pred_ids = torch.argmax(logits, dim=-1).detach().numpy()
#     corr = 0
#     total = 0
#     win_corr = 0
#     for i in range(input_ids.shape[0]):
#         if CUDA:
#             ids = np.where(b["attention_mask"][i].detach().cpu().numpy() == 1)[0]
#         else:
#             ids = np.where(b["attention_mask"][i].detach().numpy() == 1)[0]
#         _actual = input_ids[i][ids]
#         _pred = pred_ids[i][ids]

#         corr += sum(_actual[:-1] == _pred[1:])
#         win_corr += int(_actual[-1] == _pred[-2])
#         total += len(ids)

#     return corr/total, win_corr/len(input_ids)

    


# ---- Legacy ---- #

# class DataLoader(object):
#     def __init__(self, lm_fpath, res_fpath, move_to_id_fpath, maxlen, buffer_size=1e4, batch_size=4028, train_val_split=0.1, upto=-1):
#         """
#         Main dataloader iterator object

#         :param lm_fpath: file path for integers dump file
#         :param res_fpath: file path for results dump file
#         :param move_to_id_fpath: file path for moves2id json file
#         :param maxlen: maximum length of sequence
#         :param train_val_split: ratio for validation dataset size to total dataset
#         """
#         self.train_val_split = train_val_split
#         st = time.time()
#         with open(move_to_id_fpath, "r") as m:
#             self.m2id = json.load(m)
#         print(f"⏳ Loading complete took {time.time() - st}s")

#         # self._update_m2id("[result]", max(self.m2id.values()) + 1)
#         # self._update_m2id("[pad]", max(self.m2id.values()) + 1)

#         # the dataset file is too big to load in one go so need to make a iterative reader/parser
#         self.lm_path = lm_fpath
#         self.res_fpath = res_fpath
#         self.batch_size = batch_size
#         self.maxlen = maxlen
#         self.buffer_size = buffer_size

#         # self.parse_and_store(maxlen)
#         # self.set_train_mode(True)

#     def _update_m2id(self, key, value):
#         if key not in self.m2id:
#             self.m2id.update({key: value})

#     @staticmethod
#     def _rolling_window(a, window_size):
#         shape = a.shape[:-1] + (a.shape[-1] - window_size + 1, window_size)
#         strides = a.strides + (a.strides[-1],)
#         return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

#     @staticmethod
#     def _sliding_buckets(x, s):
#         # return buckets of size seqlen
#         return [x[i*s:(i+1)*s] for i in range((len(x) // s) + min(len(x) % s, 1))]

#     def parse(self, lm, res, maxlen):
#         lm = list(map(lambda x: int(x.strip()), lm.split()))
#         lmlen = len(lm)
#         res = str(res.strip())

#         if len(lm) < maxlen - 2:
#             lm = lm + [self.m2id["[result]"],
#                        self.m2id.get(res), ] + [self.m2id["[pad]"], ]*(maxlen-lmlen-2)
#             am = [1, ]*lmlen + [0, ]*(maxlen-lmlen)
#             out = [lm]
#             am = [am]

#         else:
#             # go over each model for result thingy
#             lmstacked = self._rolling_window(np.asarray(
#                 lm), maxlen - 2)  # [lm[i] for i in idx]
#             am = [[1, ]*maxlen, ]*len(lmstacked)

#             multipier = np.asarray([0, ] + [-int(i % 2 == 1) for i in range(len(lmstacked) - 1)]) + \
#                 np.asarray([1, ] + [int(i % 2 != 1)
#                                     for i in range(len(lmstacked) - 1)])
#             multipier *= int(res)
#             multipier = list(map(lambda x: self.m2id[str(x)], multipier))
#             out = np.vstack((
#                 lmstacked.T, np.ones(len(multipier), dtype=int) *
#                 self.m2id["[result]"], multipier
#             )).T.tolist()

#         return out, am

#     def __len__(self):
#         self.total_lines = None
#         if self.total_lines is None:
#             self.total_lines = 0
#             with open(self.lm_path, "r") as f:
#                 for _ in f:
#                     self.total_lines += 1

#         return (self.total_lines // self.batch_size) + min(self.total_lines % self.batch_size, 1)

#     def __iter__(self):
#         with open(self.lm_path, "r") as lm, open(self.res_fpath, "r") as res:
#             padded_lm = []
#             attentions = []
#             for _lm, _res in zip(lm, res):
#                 if not _lm:
#                     continue
#                 _lm, _attention_mask = self.parse(_lm, _res, self.maxlen)

#                 padded_lm.extend(_lm)
#                 attentions.extend(_attention_mask)

#                 if len(padded_lm) > self.buffer_size:
#                     idx = np.arange(len(padded_lm))
#                     np.random.shuffle(idx)
#                     padded_lm = np.asarray(padded_lm)[idx]
#                     attentions = np.asarray(attentions)[idx]
#                     while (len(padded_lm) > self.batch_size):
#                         if CUDA:
#                             out = {
#                                 "input_ids": torch.from_numpy(np.asarray(padded_lm[:self.batch_size])).long().cuda(),
#                                 "attention_mask": torch.from_numpy(np.asarray(attentions[:self.batch_size])).long().cuda()
#                             }
#                         else:
#                             out = {
#                                 "input_ids": torch.from_numpy(np.asarray(padded_lm[:self.batch_size])).long(),
#                                 "attention_mask": torch.from_numpy(np.asarray(attentions[:self.batch_size])).long()
#                             }

#                         del padded_lm[:self.batch_size]
#                         del attentions[:self.batch_size]
#                         yield out
