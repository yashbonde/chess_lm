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
import numpy as np

from types import SimpleNamespace

from game import Player, self_play_one_game

# this file should have the values for AWS configuration
with open("secret.json", "r") as f:
    SECRET = SimpleNamespace(**json.load(f))

class SelfPlayManager:
    secret = SECRET    
    def __init___(self, config, vocab, inv_vocab):
        self.config = config
        self.vocab = vocab
        self.inv_vocab = inv_vocab

        # Define the players
        self._m1 = Player()
        pass

    def play_one_game(self):
        config = self.config
        self_play_one_game(
            m1,
            m2,
            vocab,
            inv_vocab,
            replay_buffer=None,
            max_moves=config.max_moves,
            depth=config.depth,
            sims=config.sims
        )
        pass

    def upload_dump(self):
        """
        this function first is supposed to pickle the new dump then update the meta of
        the dump.
        """
        pass


class SelfPlayConfig:
    max_moves = None
    depth = None
    sims = None

    def __init__(self, **kwargs):
        self.attrs = []
        for k,v in kwargs.items():
            setattr(self, k, v)
            self.attrs.append(k)

    def __repr__(self):
        return "<SelfPlayConfig max_moves = >"
