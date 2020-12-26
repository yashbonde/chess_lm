"""
This file deals with the generation of large scale self-play data. It turns out that I am facing
the same problem that Deepmind faced which is that good value prediction is hard and supervised
training collapses on ~0.8 (MSE of entire dataset and 0.). This is why they included the value
head in self play games and this is what I am going to do.

Jesus Christ Jayanti ki Shubhkamnaye!
25.12.2020 - @yashbonde
"""

import os
import json
import boto3
import numpy as np
from uuid import uuid4
from types import SimpleNamespace
from argparse import ArgumentParser

from game import self_play_one_game
from model import ModelConfig, BetaChess

import torch

BUCKET_NAME = "chess-lm-bucket"
BUCKET = boto3.resource("s3").Bucket(BUCKET_NAME)


def learn_by_self_play(model, model_ckpt_path, num_games, train_every, buffer_size, tournament_size, vocab, inv_vocab, trainer_config):
    """
    this method takes in a model, makes a dataset from competing with each other over a certain
    number of steps and then trains once the buffer is full and performs a tournament between the
    players to idenitify the best model.

    Returns:
        best_model_path: path to check point the best model
    """
    buffer = []
    trainer = SelfPlayTrainer(trainer_config)
    best_model_path = None
    for i in range(num_games):
        self_play_one_game(model, model, vocab, inv_vocab, buffer, max_moves = 10, depth = 3, sims = 2)

        # free up memory
        if len(buffer) > buffer_size:
            # keep the latest samples for training
            del buffer[:len(buffer) - buffer_size]

        if i and i % train_every == 0:
            # section where we train the model --> convert the buffer into a torch.data.Dataset for feeding
            # into the SelfPlayTrainer object directly

            model.train()
            trainer.train(buffer) # train on this buffer
            latest_ckpt_path = trainer.save_checkpoint() # save after training
            model.eval()

            # now the training is complete so perform a tournament and update the best model
            best_model_path = model_ckpt_path if best_model_path == None else best_model_path
            best_model = BaseHFGPT(model.config)
            best_model = best_model.load_state_dict(torch.load(best_model_path, map_location = "cpu"))
            new_model_win = 0
            for t in range(tournament_size):
                # for ~50% of cases play as white and other times play as black
                if t%2 == 0:
                    m1 = model
                    m2 = best_model
                    win_col = "white"
                else:
                    m1 = best_model
                    m2 = model
                    win_col = "black"
                col, res = self_play_one_game(m1, m2, vocab, inv_vocab, None, max_moves = 10, depth = 3, sims = 2)
                if res == "win" and col == win_col:
                    # new model won
                    new_model_win += 1

            # if the new player wins 55% of the games then update the path
            if new_model_win / tournament_size > 0.55:
                best_model_path = latest_ckpt_path

    return best_model_path


class SelfPlayTrainer:
    def __init__(self, config):
        self.config = config
        self.best_model = None  # best model till now

    def train(self, buffer, model):
        # train a model on the latest data from buffer
        model.train()
        pass



class SelfPlayTrainerConfig:
    buffer_size = None


class SelfPlayManager:
    bucket = BUCKET # assign the global bucket
    def __init___(self, config, vocab, inv_vocab, best_model_config):
        """
        when training we always train self._m2 model for ease
        """
        self.config = config
        self.vocab = vocab
        self.inv_vocab = inv_vocab

        # Define the players
        self._m1 = BetaChess(best_model_config)
        self._m2 = BetaChess(best_model_config)

        # load the initial models and set to eval mode
        print("Loading initial models from checkpoint:", best_model_config.model_path)
        self._m1.load_state_dict(torch.load(best_model_config.model_path))
        self._m2.load_state_dict(torch.load(best_model_config.model_path))
        self._m1.eval()
        self._m2.eval()

        # class op vars
        self.buffer_name = str(uuid4())
        self.buffer = []
        self.game_counter = 0 # keeps a counter for all the games played till now

    def upload_buffer(self):
        """
        this function first is supposed to pickle the new dump then update the meta of
        the dump.
        """
        print("Uploading buffer")
        pass

    def update_champion_model(self):
        # this function updates the weights etc. from _m2 and applies to _m1
        pass

    def start(self):
        config = self.config
        try:
            while True:
                # step 1 COLLECTION: collect data by playing a tonne of games with self
                for _ in range(config.n_data_collection_games):
                    _, _, moves = self_play_one_game(
                        m1=self._m1,
                        m2=self._m2,
                        game_id=self.game_counter,
                        vocab=self.vocab,
                        inv_vocab=self.inv_vocab,
                        replay_buffer=None,
                        max_moves=config.max_moves,
                        depth=config.depth,
                        sims=config.sims
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
                # contestent.
                self.trainer.train(self.buffer, self._m2)

                # step 3 EVALUAION: evaluate the model with current best model
                # if the contestent is better than champion
                self._m2.eval()
                new_model_win = 0
                for i in range(config.n_evaluation_games):
                    # in 50% of the games are played as white and 50% as black
                    m1 = self._m1 if i % 2 == 0 else self._m2
                    m2 = self._m2 if i % 2 == 0 else self._m1
                    win_col = "white" if i % 2 == 0 else "black"
                    col, res, _ = self_play_one_game(
                        m1=m1,
                        m2=m2,
                        game_id=self.game_counter,
                        vocab=self.vocab,
                        inv_vocab=self.inv_vocab,
                        replay_buffer=None,
                        max_moves=config.max_moves,
                        depth=config.depth,
                        sims=config.sims
                    )
                    if res == "win" and col == win_col:
                        # new model won
                        new_model_win += 1
                
                if new_model_win / config.n_evaluation_games >= 0.55:
                    self.update_champion_model()

                
        except KeyboardInterrupt:
            print("Found KeyboardInterrupt, stopping training and gameplay collection")
            self.upload_buffer()


class SelfPlayConfig:
    max_moves = None
    depth = None
    sims = None
    buffer_size = None
    n_data_collection_games = None # number of games to play for collection

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
    args.add_argument("--m2id", type=str, default = "assets/moves.json", help="path to move_to_id json")
    args.add_argument("--max_moves", type=int, default = 180, help="path to move_to_id json")
    args.add_argument("--depth", type=int, default = 80, help="path to move_to_id json")
    args.add_argument("--sims", type=int, default = 10, help="path to move_to_id json")
    args.add_argument("--buffer_size", type=int, default = 10000, help="path to move_to_id json")
    args = args.parse_args()

    # load vocab
    with open(args.m2id, "r") as f:
        vocab = json.load(f)
        inv_vocab = {v:k for k,v in vocab.items()}

    # best model architecture configuration till 26.12.2020
    best_model_config = ModelConfig(
        vocab_size=len(vocab),
        n_positions=85*2,
        n_ctx=85*2,
        n_embd=200,
        n_layer=10,
        n_head=20,
        loss_method="mse", # simple regression head
        vocab_path = args.m2id,
        model_path = args.best_model_path
    )

    # config for SelfPlay
    selfplay_config = SelfPlayConfig(
        max_moves=args.max_moves,
        depth=args.depth,
        sims=args.sims,
        buffer_size=args.buffer_size,
    )

    manager = SelfPlayManager(
        config=selfplay_config,
        vocab=vocab,
        inv_vocab=inv_vocab,
        best_model_config=best_model_config
    )
