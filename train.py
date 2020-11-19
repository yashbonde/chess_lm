# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Yash Bonde
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
from torch import nn
from torch.utils import tensorboard as tb

from torch.utils.data import DataLoader
from model import DataConfig, ChessData, ModelConfig, BaseHFGPT, TrainerConfig, Trainer



# load user args
args = ArgumentParser(description="Train GPT2 model on t2sql corpus")
args.add_argument(
    "--train",
    type=int,
    choices=[0, 1],
    help="Either to train the model from scratch (train) or finetune (fine)",
    default = 1
)

# data args
args.add_argument("--lmtrain", type=str, default = None, help="path to train_lm file")
args.add_argument("--lmtest", type=str, default = None, help="path to test_lm file")
args.add_argument("--res", type=str, default = None, help="path to res file")
args.add_argument("--m2id", type=str, default = "m2id.json", help="path to move_to_id json")

args.add_argument("--maxlen", type = int, default = 60, help = "maximum length")
args.add_argument("--buffer", type = int, default = 99999, help = "buffer size for DataSet")
args.add_argument("--n_embd", type = int, default = 128, help = "embedding dim")
args.add_argument("--n_layer", type = int, default = 30, help = "number of layers of the model")
args.add_argument("--n_head", type = int, default = 8, help = "number of heads for MHA")

args.add_argument("--lr", type = int, default = 0.0001, help = "learning rate")
args.add_argument("--beta1", type = int, default = 0.9, help = "Adam.beta1")
args.add_argument("--beta2", type = int, default = 0.95, help = "Adam.beta2")

# train args
args.add_argument("--batch_size", type=int, default=350, help="batch size")
args.add_argument("--num_epochs", type=int, default=1, help="Number of epochs to train / finetune")
args.add_argument("--train_val_split", type=float, default=0.01, help="Ratio for validation dataset size to total dataset")
args.add_argument("--save_folder", type=str, default="models", help="Folder to save model to")
args.add_argument("--model", type=str, default="cgpt", help="Saved model to have filepath `<model>.pt`")
args.add_argument("--save_every", type=int, default=1000, help="Save model every this global steps")
args = args.parse_args()

# path and file management
os.makedirs(args.save_folder, exist_ok=True)  # make folder if needed
model_folder = os.path.join(args.save_folder, args.model)
os.makedirs(args.save_folder, exist_ok=True)  # make folder if needed
model_path = os.path.join(model_folder, args.model + ".pt")

dataConfig = DataConfig(
    lm=args.lmtrain,
    m2id=args.m2id,
    maxlen=args.maxlen,
    buffer= args.buffer,
)
dstrain = ChessData(dataConfig)

# dataConfig = DataConfig(
#     lm=args.lmtest,
#     m2id=args.m2id,
#     maxlen=args.maxlen,
#     buffer= args.buffer,
# )
# dstest = ChessData(dataConfig)

modelConfig = ModelConfig(
    vocab_size = len(dstrain.m2id),
    n_positions=args.maxlen,
    n_ctx=args.maxlen,
    n_embd=args.n_embd,
    n_layer=args.n_layer,
    n_head=args.n_head
)
print(modelConfig)
model = BaseHFGPT(modelConfig)

trainerConf = TrainerConfig(
    max_epochs = args.num_epochs,
    batch_size = args.batch_size,
    lr = args.lr,
    betas = (args.beta1, args.beta2),
    tb_path = model_folder
)
trainer = Trainer(model, dstrain, trainerConf)

# if user is going to finetune an existing model
if args.train == 0:
    assert os.path.exists(model_path), f"Model not found on path: {model_path}"
    print(f"🔋 Finetuning model at: {model_path}")
    model.load_state_dict(torch.load(model_path))
elif args.train == 1:
    print(f"🔪 Training a new model")

print(f"Model Size: {sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())}")

# train
trainer.train()
