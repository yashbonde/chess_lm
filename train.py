# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Finetuning GPT2 to play chess
@yashbonde-09.08.2020"""

from collections import deque
import os
import json
import time
import numpy as np
from tqdm import trange
from argparse import ArgumentParser

import torch
from torch.utils import tensorboard as tb
from transformers import GPT2Config, GPT2LMHeadModel

CUDA = bool(torch.cuda.device_count())

class DataLoader(object):
    def __init__(self, lm_fpath, res_fpath, move_to_id_fpath, maxlen, buffer_size = 1e4, batch_size=4028, train_val_split=0.1, upto=-1):
        """
        Main dataloader iterator object

        :param lm_fpath: file path for integers dump file
        :param res_fpath: file path for results dump file
        :param move_to_id_fpath: file path for moves2id json file
        :param maxlen: maximum length of sequence
        :param train_val_split: ratio for validation dataset size to total dataset
        """
        self.train_val_split = train_val_split
        st = time.time()
        with open(move_to_id_fpath, "r") as m:
            self.m2id = json.load(m)
        print(f"⏳ Loading complete took {time.time() - st}s")

        # self._update_m2id("[result]", max(self.m2id.values()) + 1)
        # self._update_m2id("[pad]", max(self.m2id.values()) + 1)

        # the dataset file is too big to load in one go so need to make a iterative reader/parser
        self.lm_path = lm_fpath
        self.res_fpath = res_fpath        
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.buffer_size = buffer_size

        # self.parse_and_store(maxlen)
        # self.set_train_mode(True)

    def _update_m2id(self, key, value):
        if key not in self.m2id:
            self.m2id.update({key: value})

    @staticmethod
    def _rolling_window(a, window_size):
        shape = a.shape[:-1] + (a.shape[-1] - window_size + 1, window_size)
        strides = a.strides + (a.strides[-1],)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

    @staticmethod
    def _sliding_buckets(x, s):
        # return buckets of size seqlen
        return [x[i*s:(i+1)*s] for i in range((len(x) // s) + min(len(x) % s, 1))]

    def parse(self, lm, res, maxlen):
        lm = list(map(lambda x: int(x.strip()), lm.split()))
        lmlen = len(lm)
        res = str(res.strip())

        if len(lm) < maxlen - 2:
            lm = lm + [self.m2id["[result]"], self.m2id.get(res),] + [self.m2id["[pad]"], ]*(maxlen-lmlen-2)
            am = [1, ]*lmlen + [0, ]*(maxlen-lmlen)
            out = [lm]
            am = [am]

        else:
            # go over each model for result thingy
            lmstacked = self._rolling_window(np.asarray(lm), maxlen - 2)  # [lm[i] for i in idx]
            am = [[1,]*maxlen,]*len(lmstacked)

            multipier = np.asarray([0, ] + [-int(i % 2 == 1) for i in range(len(lmstacked) - 1)]) + \
                np.asarray([1, ] + [int(i % 2 != 1) for i in range(len(lmstacked) - 1)])
            multipier *= int(res)
            multipier = list(map(lambda x: self.m2id[str(x)], multipier))
            out = np.vstack((
                lmstacked.T, np.ones(len(multipier), dtype=int) *
                self.m2id["[result]"], multipier
            )).T.tolist()
        
        return out, am

    # def parse_and_store(self, maxlen):
    #     data = []
    #     attentions = []
    #     pbar = trange(len(self.lms))
    #     # for idx, lm in enumerate(self.lms):
    #     for idx in pbar:
    #         pbar.set_description( f"parsing {idx + 1}, {((idx + 1)/len(self.lms)*100)}% done!")
    #         lm = self.lms[idx]
    #         lmlen = len(lm)

    #         if len(lm) < maxlen - 2:
    #             lm = (
    #                 self.lms[idx] +
    #                 [
    #                     self.m2id["[result]"],
    #                     self.m2id.get(str(self.results[idx])),
    #                 ] +
    #                 [self.m2id["[pad]"], ]*(maxlen-lmlen-2))
    #             am = [1, ]*lmlen + [0, ]*(maxlen-lmlen)
    #             out = [lm]
    #             am = [am]

    #         else:
    #             # go over each model for result thingy
    #             lmstacked = self._rolling_window(np.asarray(lm), maxlen - 2)  # [lm[i] for i in idx]
    #             am = [[1,]*maxlen,]*len(lmstacked)

    #             multipier = np.asarray([0, ] + [-int(i % 2 == 1) for i in range(len(lmstacked) - 1)]) + \
    #                 np.asarray([1, ] + [int(i % 2 != 1) for i in range(len(lmstacked) - 1)])
    #             multipier *= self.results[idx]
    #             multipier = list(map(lambda x: self.m2id[str(x)], multipier))
    #             out = np.vstack((
    #                 lmstacked.T, np.ones(len(multipier), dtype=int) *
    #                 self.m2id["[result]"], multipier
    #             )).T.tolist()
            
    #         data.extend(out)
    #         attentions.extend(am)

    #     self.idx = np.arange(len(self.lms))
    #     np.random.shuffle(self.idx)
    #     split_idx = int(self.idx.shape[0] * self.train_val_split)

    #     self.val_dataset = np.asarray(data[:split_idx])
    #     self.val_attention_masks = np.asarray(attentions[:split_idx])
    #     self.train_attention_masks = np.asarray(attentions[split_idx:])
    #     self.train_dataset = np.asarray(data[split_idx:])

    # def set_train_mode(self, verbose = False):
    #     if verbose:
    #         print("💡 Using TRAIN dataset")
    #     self.data = self.train_dataset
    #     self.attention_mask = self.train_attention_masks

    # def set_val_mode(self, verbose = False):
    #     if verbose:
    #         print("💡 Using VALIDATION dataset")
    #     self.data = self.val_dataset
    #     self.attention_mask = self.val_attention_masks

    def __len__(self):
        self.total_lines = None
        if self.total_lines is None:
            self.total_lines = 0
            with open(self.lm_path, "r") as f:
                for _ in f:
                    self.total_lines += 1

        return (self.total_lines // self.batch_size) + min(self.total_lines % self.batch_size, 1)

    def __iter__(self):
        with open(self.lm_path, "r") as lm, open(self.res_fpath, "r") as res:
            padded_lm = []
            attentions = []
            for _lm, _res in zip(lm, res):
                if not _lm:
                    continue
                _lm, _attention_mask = self.parse(_lm, _res, self.maxlen)
                
                padded_lm.extend(_lm)
                attentions.extend(_attention_mask)

                if len(padded_lm) > self.buffer_size:
                    idx = np.arange(len(padded_lm))
                    np.random.shuffle(idx)
                    padded_lm = np.asarray(padded_lm)[idx].tolist()
                    attentions = np.asarray(attentions)[idx].tolist()
                    if CUDA:
                        out = {
                            "input_ids": torch.from_numpy(np.asarray(padded_lm[:self.batch_size])).long().cuda(),
                            "attention_mask": torch.from_numpy(np.asarray(attentions[:self.batch_size])).long().cuda()
                        }
                    else:
                        out = {
                            "input_ids": torch.from_numpy(np.asarray(padded_lm[:self.batch_size])).long(),
                            "attention_mask": torch.from_numpy(np.asarray(attentions[:self.batch_size])).long()
                        }

                    del padded_lm[:self.batch_size]
                    del attentions[:self.batch_size]

                    yield out


    # def __iter__(self):
    #     idx = np.arange(self.data.shape[0])
    #     np.random.shuffle(idx)
    #     stack = []
    #     for batched_ids in self._sliding_buckets(idx, self.batch_size):
    #         stack.append({
    #             "input_ids": torch.from_numpy(self.data[batched_ids]).long(),
    #             "attention_mask": torch.from_numpy(self.attention_mask[batched_ids]).long()
    #         })

    #     # return iterator over data
    #     return iter(stack)


def accuracy(b, logits):
    # (upto -1) compared to (from 1)
    if CUDA:
        input_ids = b["input_ids"].detach().cpu().numpy()
        pred_ids = torch.argmax(logits, dim=-1).detach().cpu().numpy()
    else:
        input_ids = b["input_ids"].detach().numpy()
        pred_ids = torch.argmax(logits, dim=-1).detach().numpy()
    corr = 0
    total = 0
    win_corr = 0
    for i in range(input_ids.shape[0]):
        if CUDA:
            ids = np.where(b["attention_mask"][i].detach().cpu().numpy() == 1)[0]
        else:
            ids = np.where(b["attention_mask"][i].detach().numpy() == 1)[0]
        _actual = input_ids[i][ids]
        _pred = pred_ids[i][ids]

        corr += sum(_actual[:-1] == _pred[1:])
        win_corr += int(_actual[-1] == _pred[-2])
        total += len(ids)

    return corr/total, win_corr/len(input_ids)


if __name__ == "__main__":
    # load user args
    args = ArgumentParser(description="Train GPT2 model on t2sql corpus")
    args.add_argument("--tf", type=str, choices=["t", "f"],
                      help="Either to train the model from scratch (t) or finetune (f)")

    # data args
    args.add_argument("--lm", type=str, required = True,
                      help="path to lm file")
    args.add_argument("--res", type=str, required = True,
                      help="path to res file")
    args.add_argument("--m2id", type=str, required=True,
                      help="path to move_to_id json")
    args.add_argument("--config", type=str, required=True,
                      help="path to move_to_id json")

    # train args
    args.add_argument("--num_epochs", type=int, default=1,
                      help="Number of epochs to train / finetune")
    args.add_argument("--train_val_split", type=float, default=0.01,
                      help="Ratio for validation dataset size to total dataset")
    args.add_argument("--save_folder", type=str,
                      default="models", help="Folder to save model to")
    args.add_argument("--model", type=str, default="cgpt",
                      help="Saved model to have filepath `<model>.pt`")
    args.add_argument("--save_every", type=int, default=1000,
                      help="Save model every this global steps")
    args.add_argument("--tensorboard", nargs = "?", type = bool, default = False)
    args = args.parse_args()

    # path and file management
    os.makedirs(args.save_folder, exist_ok=True)  # make folder if needed
    model_folder = os.path.join(args.save_folder, args.model)
    os.makedirs(args.save_folder, exist_ok=True)  # make folder if needed
    model_path = os.path.join(model_folder, args.model + ".pt")

    # load configuration
    with open(args.m2id, "r") as f, open(args.config, "r") as c:
        raw_config = json.load(c)
        # print(len(json.load(f).values()), json.load(f))
        raw_config.update({
            "vocab_size": len(json.load(f).values()),
            "n_ctx": raw_config["maxlen"],
            "n_positions": raw_config["maxlen"]
        })
        config = GPT2Config(**raw_config)
        print(config)
    model = GPT2LMHeadModel(config)
    if CUDA:
        model.cuda()

    # load dataset
    dataset = DataLoader(
        lm_fpath=args.lm,
        res_fpath=args.res,
        move_to_id_fpath=args.m2id,
        maxlen=config.maxlen,
        batch_size=config.batch_size,
        train_val_split=args.train_val_split,
        upto=-1
    )

    # if user is going to finetune an existing model
    if args.tf == "f":
        assert os.path.exists(
            model_path), f"Model not found on path: {model_path}"
        print(f"🔋 Finetuning model at: {model_path}")
        model.load_state_dict(torch.load(model_path))

    # for i in range(2):
    #     print("================ Epoch", i)
    #     for didx, d in enumerate(dataset):
    #         print(didx)

    print(f"Model Size: {sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())}")

    # now proceed to training
    global_step = 0
    val_step = 0
    optim = getattr(torch.optim, raw_config["optimizer"])(params = model.parameters(), **raw_config["optimizer_params"])
    
    summary_writer = None
    loss = -1
    if args.tensorboard:
        summary_writer = tb.SummaryWriter(log_dir=model_folder, comment="Hello World!", flush_secs=20)

    try:
        for e in range(args.num_epochs):
            # dataset.set_train_mode(False) # dep
            model.train()
            pbar = trange(len(dataset))
            for bidx, b in zip(pbar, dataset):
                if not isinstance(loss, int):
                    loss = loss.item()
                pbar.set_description(
                    f"Epoch: {e}, TRAIN, batch: {bidx}, loss: {round(loss, 3)}")
                model.zero_grad()
                out = model(**b, labels=b["input_ids"])
                loss, logits = out[:2]
                loss.backward()
                optim.step()

                total_acc, winner_acc = accuracy(b, logits)
                if args.tensorboard:
                    summary_writer.add_scalar( "Loss/Train", loss.item(), global_step, walltime=time.time())
                    summary_writer.add_scalar( "Accuracy/Train_Total", total_acc * 100, global_step, walltime=time.time())
                    summary_writer.add_scalar( "Accuracy/Train_Winner_Prediction", winner_acc * 100, global_step, walltime=time.time())

                if global_step % args.save_every == 0:
                    print(f"📀 Saving Model.... at: {model_path}")
                    torch.save(model.state_dict(), model_path)

                global_step += 1

            # dataset.set_val_mode(False)
            # model.eval()
            # pbar = trange(len(dataset))
            # for bidx, b in zip(pbar, dataset):
            #     pbar.set_description(f"Epoch: {e}, VAL, batch: {bidx}")
            #     out = model(**b, labels=b["input_ids"])
            #     loss, logits = out[:2]

            #     total_acc, winner_acc = accuracy(b, logits)
            #     if args.tensorboard:
            #         summary_writer.add_scalar("Loss/Val", loss.item(), val_step, walltime= time.time())
            #         summary_writer.add_scalar("Accuracy/Val_Total", total_acc * 100, val_step, walltime=time.time())
            #         summary_writer.add_scalar("Accuracy/Val_Winner_Prediction", winner_acc * 100, val_step, walltime=time.time())

            #     val_step += 1
    except KeyboardInterrupt as e:
        print(f"📀 Saving Model.... at: {model_path}")
        torch.save(model.state_dict(), model_path)
    if args.tensorboard:
        summary_writer.close()
