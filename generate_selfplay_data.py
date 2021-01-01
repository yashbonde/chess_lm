"""
This file deals with the generation of large scale self-play data. It turns out that I am facing
the same problem that Deepmind faced which is that good value prediction is hard and supervised
training collapses on ~0.8 (MSE of entire dataset and 0.). This is why they included the value
head in self play games and this is what I am going to do.

Jesus Christ Jayanti ki Shubhkamnaye!
25.12.2020 - @yashbonde
"""

import re
import os
import json
import math
from sys import version
import boto3
import elote
import pickle
import tarfile
import numpy as np
from uuid import uuid4
from tqdm import trange
from types import SimpleNamespace
from argparse import ArgumentParser

from game import self_play_one_game, verbose_print
from model import ModelConfig, BetaChess, FullDatasetPreLoaded, configure_optimizers

import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader

# assign the global bucket
BUCKET_NAME = "chess-lm-bucket"
BUCKET = boto3.resource("s3").Bucket(BUCKET_NAME)
FILES_ON_AWS = [obj.key for obj in BUCKET.objects.all()]
LOCAL = bool(float(os.getenv("LOCAL", False)))

class SelfPlayTrainer:
    def __init__(self, config, init_model, device):
        self.config = config
        self.device = device
        self.best_model = None  # best model till now

        # get AdamW optimiser for this model
        self.optimizer = configure_optimizers(init_model, config)
        print("Currently the system will only use a GPT-3 style scheduler")
        self.scheduler = None
        self.processed_tokens = 0  # number of tokens processed till now
        self.ggs = 0  # all the global steps, not just the local ones

    def save_checkpoint(self, ckpt_path = None, model = None):
        raw_model = model.module if hasattr(model, "module") else model
        ckpt_path = ckpt_path if ckpt_path is not None else self.config.ckpt_path
        print(f"Saving Model at {ckpt_path}")
        torch.save(raw_model.state_dict(), ckpt_path)

    def train(self, lms, results, model):
        """train a model on the latest data from buffer
        the buffer is a very long list of <BufferMovePoint> objects
        """
        model.train()
        config = self.config
        model_config = model.module.config if hasattr(model, "module") else model.config
        # step 1: shape the buffer
        train_data = FullDatasetPreLoaded(lms, results, m2id = None)
        num_batches = len(train_data) // model_config.n_ctx + int(len(train_data) % config.batch_size != 0)
        print(len(train_data), num_batches)
        pbar_train = trange(num_batches, ncols=100)
        dl_train = DataLoader(dataset=train_data, pin_memory=True, batch_size=config.batch_size, shuffle=train_data.shuffle)
        prev_train_loss = 100000 # what was the training loss in previous testing cycle
        no_loss_steps = 0 # no of steps since there was devrease in the loss
        break_training = False
        train_losses = [-1]
        train_acc = [-1]

        for gs, d in zip(pbar_train, dl_train):
            # total steps is now the primary iteration method
            d = {k:v.to(self.device) for k,v in d.items()}
            pbar_train.set_description(f"[TRAIN] GGS:{self.ggs}, GS: {gs}, Loss: {round(train_losses[-1], 5)}")

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

            # backprop
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            self.optimizer.step()
            if self.scheduler is not None and self.scheduler != "GPT3":
                last_lr = self.scheduler.get_last_lr()[0]
                log_dict.update({"lr": last_lr})
                self.scheduler.step()

            # ------------- LR SCHEDULING
            elif self.scheduler == "GPT3":
                # update learning rate
                self.processed_tokens += d["input_ids"].size(0) * model_config.n_ctx # batch_size * number of tokens in each sequence
                if self.processed_tokens < config.warmup_tokens:
                    # linear warmup
                    lr_mult = float(self.processed_tokens) / float(max(1, config.warmup_tokens))
                else:
                    # cosine learning rate decay
                    progress = float(self.processed_tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                    lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                lr = config.lr * lr_mult
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                log_dict.update({"lr": lr})
            # ------------- LR SCHEDULING

            if gs and gs % config.save_every == 0:
                cp = config.ckpt_path.replace(".pt", f"_self_{self.ggs}.pt")
                self.save_checkpoint(cp, model)

        # training loops end, save final checkpoint file and update ggs
        self.ggs += gs  # update
        print("Train End Saving")
        cp = config.ckpt_path.replace(".pt", f"_self_{self.ggs}.pt")
        self.save_checkpoint(cp, model)

class SelfPlayTrainerConfig:
    num_epochs = 2
    batch_size = 64
    lr = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    num_workers = 0 # for DataLoader
    ckpt_path = None
    tb_path = None
    patience = -1
    save_every = None
    scheduler = None
    weight_decay = 0.1
    warmup_perc = None
    warmup_tokens = None
    final_tokens = None

    def __init__(self, **kwargs):
        self.attrs = [
            "num_epochs", "batch_size", "lr", "betas", "grad_norm_clip", "num_workers",
            "ckpt_path", "tb_path", "patience", "save_every", "scheduler", "weight_decay",
            "warmup_perc", "warmup_tokens", "final_tokens",
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
            self.final_tokens = 613256130  # total size of all the tokens
            self.warmup_tokens = int(self.final_tokens * self.warmup_perc)
            print("Auto Setting warmup_tokens using", self.warmup_perc, "to", self.warmup_tokens)

        elif self.scheduler == "GPT3":
            assert self.final_tokens != None
            assert self.warmup_tokens != None

    def __repr__(self):
        return "---- SELFPLAY TRAINER CONFIGURATION ----\n" + \
            "\n".join([f"{k}\t{getattr(self, k)}" for k in list(set(self.attrs))
        ]) + "\n"


class SelfPlayManager:
    def __init__(self, config, vocab, inv_vocab, best_model_config, trainer_config, m1_elo = 1400, m2_elo = 1400, verbose = False):
        """
        when training we always train self._m2 model for ease and copy weights to self._m1 model
        """
        self.config = config
        self.vocab = vocab
        self.inv_vocab = inv_vocab
        self.verbose = verbose

        # Define the players
        self._m1 = BetaChess(best_model_config)
        self._m2 = BetaChess(best_model_config)

        # load the initial models and set to eval mode
        self.device = "cpu" if not torch.cuda.is_available() else f"cuda:{torch.cuda.current_device()}"
        print("Loading initial models from checkpoint:", best_model_config.model_path, "::: to device:", self.device)
        self._m1.load_state_dict(torch.load(best_model_config.model_path, map_location=self.device))
        self._m2.load_state_dict(torch.load(best_model_config.model_path, map_location=self.device))
        self._m1.eval()
        self._m2.eval()
        self._m1_elo = m1_elo # initial ELO rating
        self._m2_elo = m2_elo # initial ELO rating
        self.model_config = self._m1.config

        if "cuda" in self.device:
            # now paralelize
            self._m1 = torch.nn.DataParallel(self._m1).to(self.device)
            self._m2 = torch.nn.DataParallel(self._m2).to(self.device)
            print("Model is now CUDA!")

        # load the trainer object
        self.trainer = SelfPlayTrainer(trainer_config, self._m2, self.device)

        # class op vars
        self.buffer_name = str(uuid4())
        self.buffer = []
        self.game_counter = 0 # keeps a counter for all the games played till now

    def upload_run(self, upto_idx = None):
        """
        this function first is supposed to pickle the new dump then update the meta of
        the dump.
        """
        print("-"*70)
        print("UPLOADING BUFFER")
        print("Total Buffer Size:", len(self.buffer))
        gids = [re.findall(r"\d+", x) for x in FILES_ON_AWS if "dump" in x]
        max_game_id = 0
        for g in gids:
            max_game_id = max(max(g), max_game_id)

        fname = f"./dump_{max_game_id+1}_{max_game_id+self.game_counter}.pkl"
        print("Target pickle object:", fname)
        with open(fname, "wb") as pkl_file:
            if upto_idx:
                pickle.dump(self.buffer[:upto_idx], pkl_file)
            else:
                pickle.dump(self.buffer, pkl_file)
        tar_fname = fname[:-3] + "tar.gz"
        print("Compressing to:", tar_fname)
        with tarfile.open(tar_fname, "w:gz") as tar:
            tar.add(fname, arcname=os.path.basename(fname))
        print(f"Uploading buffer {tar_fname} ...")
        if LOCAL:
            print("Demo.. no upload because found envvar LOCAL = True!")
        else:
            BUCKET.upload_file(fname, fname) # local, cloud
        print(" ... Completed Upload! Flushing local buffer")
        if upto_idx is not None:
            del self.buffer[:upto_idx]
        # no else because the script is exitng anyways
        print("-"*70)


    def update_elo(self, wins, draw, losts):
        # we are using BayesElo as proposed here and used in Alpha/Mu Zero for Chess
        # and Shogi. Read more about it here:
        # https://www.remi-coulom.fr/Bayesian-Elo/
        # basically the algorithm looks like this:
        # probability that A defeats B
        # f(x) = 1 / (1 + 10 ** (x / 400))
        # p(WhiteWins) = f(eloBlack - eloWhite - eloAdvantage + eloDraw)
        # p(BlackWins) = f(eloWhite - eloBlack + eloAdvantage + eloDraw)
        # p(draw)      = 1 - p(WhiteWins) - p(BlackWins)
        # eloAdvantage = 32.8 +/- 4
        # eloDraw      = 97.3 +/- 2
        #
        # now despite all this I do not know how to calculate this. So I am using
        # package called elote, the code is borrowed from:
        # https://github.com/peldszus/alpha-zero-general-lib/blob/master/src/alpha_zero_general/league.py
        a = elote.EloCompetitor(self._m1_elo)
        b = elote.EloCompetitor(self._m2_elo)
        rating_a = a.rating
        rating_b = b.rating
        verbose_print("[BEFORE] Rating A:", self._m1_elo, "Rating B:", self._m2_elo, verbose=self.verbose)
        verbose_print(f"wins: {wins}; draw: {draw}; losts: {losts}", verbose=self.verbose)
        if wins > losts:
            a.beat(b)
        elif losts > wins:
            b.beat(a)
        else:
            a.tied(b)
        rating_change_a = a.rating - rating_a
        rating_change_b = b.rating - rating_b
        self._m1_elo = a.rating
        self._m2_elo = b.rating
        verbose_print("[AFTER] Rating A:", self._m1_elo, "Rating B:", self._m2_elo, verbose=self.verbose)
        return rating_change_a, rating_change_b

    def update_champion_model(self):
        # this function updates the weights etc. from _m2 and puts them in _m1
        # usually this is very tricky when using optimizer etc. but since _m1
        # is guaranteed to be used for inference only we can get away with it.
        verbose_print("Updating _m1 weights from _m2", verbose=self.verbose)
        self._m1.load_state_dict(self._m2.state_dict())

    def start(self):
        config = self.config
        try:
            while True:
                # step 1 COLLECTION: collect data by playing a tonne of games with self
                verbose_print("Perform data collection by self play", verbose=self.verbose)
                pbar = trange(config.n_data_collection_games)
                for i in pbar:
                    # 50 % of the games are played with white as winning color
                    # and rest with black as winning color
                    m1 = self._m1 if i % 2 == 0 else self._m2
                    m2 = self._m2 if i % 2 == 0 else self._m1
                    win_col = "black" if i % 2 == 0 else "white" # color for m2
                    pbar.set_description(f"[SELF PLAY] win color: {win_col}")
                    _, _, moves = self_play_one_game(
                        m1=m1,
                        m2=m2,
                        win_col=win_col,
                        game_id=self.game_counter,
                        vocab=self.vocab,
                        inv_vocab=self.inv_vocab,
                        replay_buffer=None,
                        max_moves=config.max_moves,
                        depth=config.depth,
                        sims=config.sims,
                        _trange_moves=self.verbose,
                        _CUDA="cuda" in self.device
                    )
                    self.game_counter += 1
                    self.buffer.extend(moves)

                    # free up memory
                    if len(self.buffer) > config.buffer_size:
                        # keep the latest samples for training
                        del self.buffer[:len(self.buffer) - config.buffer_size]

                # step 2 OPTIMISATION: train the model
                # note that in this approach we always train the second model for convinience
                # _m1 is champion and _m2 is a contestent this is why always train the
                # contestent. For this we need to define the lms and results array for
                # training
                buffer = np.array([(x.move_id, x.value) for x in self.buffer])
                lms, results = buffer[:, 0], buffer[:, 1]
                upto_idx = -(len(lms) % self.model_config.n_ctx)
                lms = np.array(lms[:upto_idx]).reshape(-1, self.model_config.n_ctx)
                results = np.array(results[:upto_idx]).reshape(-1, self.model_config.n_ctx)
                verbose_print("lms:", lms.shape,";Results:", results.shape, verbose = self.verbose)
                self.trainer.train(lms, results, self._m2)

                # step 3 EVALUAION: evaluate the model with current best model
                # if the contestent is better than champion
                verbose_print("Perform the tournament of the models", verbose=self.verbose)
                self._m2.eval()
                new_model_win = 0
                num_draws = 0
                pbar = trange(config.n_evaluation_games)
                for i in pbar:
                    # in 50% of the games are played as white and 50% as black
                    m1 = self._m1 if i % 2 == 0 else self._m2
                    m2 = self._m2 if i % 2 == 0 else self._m1
                    win_col = "black" if i % 2 == 0 else "white" # color for m2
                    pbar.set_description(f"[EVLUATION] win color: {win_col}")
                    col, res, _ = self_play_one_game(
                        m1=m1,
                        m2=m2,
                        win_col=win_col,
                        game_id=self.game_counter,
                        vocab=self.vocab,
                        inv_vocab=self.inv_vocab,
                        replay_buffer=None,
                        max_moves=config.max_moves,
                        depth=config.depth,
                        sims=config.sims,
                        _trange=self.verbose,
                        _CUDA="cuda" in self.device
                    )
                    if res == "win" and col == win_col:
                        # new model won
                        new_model_win += 1
                    elif res in ["draw", "game"]:
                        # either the game was draw or did not finish
                        num_draws += 1
                
                new_model_lost = config.n_evaluation_games - num_draws - new_model_win
                s = new_model_win / config.n_evaluation_games
                verbose_print("Score:", s, verbose=self.verbose)
                if s >= 0.55:
                    self.update_champion_model()

                # calculate the new ELOs
                m1_change, m2_change = self.update_elo(new_model_win, num_draws, new_model_lost)

                # upload the new datasets and delete local buffer till upto_idx
                self.upload_run(upto_idx)
                
        except KeyboardInterrupt:
            print("Found KeyboardInterrupt, stopping training and gameplay collection")
            self.upload_run()


class SelfPlayConfig:
    max_moves = None
    depth = None
    sims = None
    buffer_size = None
    n_data_collection_games = None # number of games to play for collection
    n_evaluation_games = None      # number of games to play for evaluation

    def __init__(self, **kwargs):
        self.attrs = ["max_moves", "depth", "sims", "buffer_size",
                      "n_data_collection_games"]
        for k,v in kwargs.items():
            setattr(self, k, v)
            self.attrs.append(k)

    def __repr__(self):
        return "---- SELF PLAY CONFIGURATION ----\n" + \
            "\n".join([f"{k}\t{getattr(self, k)}" for k in list(set(self.attrs))
        ]) + "\n"


if __name__ == "__main__":
    args = ArgumentParser(description="Generate self play data and store in AWS. "
        "If you want to change the model configuration you want to use, please "
        "this script manually. This script *assumes* that you have a working `awscli` "
        "with `aws_access_key_id` and `aws_secret_access_keykeys`. Since this uses "
        "boto3 it will automatically load the data from there. Please open this "
        "file and update BUCKET_NAME. Happy Hunting!"
    )
    args.add_argument("--best_model_path", type=str, required = True, help="path to checkpoint file to best model")
    args.add_argument("--arch", type=str, choices=["tiny", "medium"], default = "tiny", help="architecture")
    args.add_argument("--m2id", type=str, default = "assets/moves.json", help="path to move_to_id json")
    args.add_argument("--max_moves", type=int, default = 100, help="number of moves to play in the game")
    args.add_argument("--depth", type=int, default = 5, help="max tree depth in recursion for MCTS")
    args.add_argument("--sims", type=int, default = 100, help="number of simulations to perform for each move")
    args.add_argument("--buffer_size", type=int, default = int(1e9), help="total training buffer size")
    args.add_argument("--n_data_collection_games", type=int, default = 100, help="total games to perform in each training loop for data collection")
    args.add_argument("--n_evaluation_games", type=int, default = 4, help="number of games for evaluation")
    args = args.parse_args()

    # load vocab
    with open(args.m2id, "r") as f:
        vocab = json.load(f)
        inv_vocab = {v:k for k,v in vocab.items()}

    if argts.arch == "tiny":
        n_embd = 128
        n_layer = 8
        n_head = 8
    else: # medium
        n_embd = 200
        n_layer = 10
        n_head = 10

    best_model_config = ModelConfig(
        vocab_size=len(vocab),
        n_positions=85*2,
        n_ctx=85*2,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        loss_method="mse", # simple regression head
        vocab_path = args.m2id,
        model_path = args.best_model_path
    )
    print(best_model_config)

    # config for SelfPlay
    selfplay_config = SelfPlayConfig(
        depth=args.depth,
        sims=args.sims,
        buffer_size=args.buffer_size,
        max_moves = args.max_moves,
        n_data_collection_games=args.n_data_collection_games,
        n_evaluation_games=args.n_evaluation_games,
    )
    print(selfplay_config)

    # config for trainer
    trainer_config = SelfPlayTrainerConfig(
        num_epochs=2,
        batch_size=64,
        lr=3e-4,
        betas=(0.9, 0.95),
        grad_norm_clip=1.0,
        ckpt_path=args.best_model_path, # no overrides happen
        save_every=100,
        scheduler="GPT-3", # default, right now we only support GPT-3 style
        weight_decay=0.1,
        warmup_perc=0.1,
        warmup_tokens=100,
        final_tokens=10000,
    )
    print(trainer_config)

    # define the manger
    manager = SelfPlayManager(
        config=selfplay_config,
        vocab=vocab,
        trainer_config = trainer_config,
        inv_vocab=inv_vocab,
        best_model_config=best_model_config,
        verbose = True
    )
    print(manager)

    # train from the manager
    manager.start()
