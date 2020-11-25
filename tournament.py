"""run the models and calculate ELO ratings
19.11.2020 - @yashbonde"""

from argparse import ArgumentParser
from model import ModelConfig
from game import Player

import torch


def expected(p1, p2):
    return 1 / (1 - 10 ** ((p2 - p1) / 400))


def elo(p, e, s, k=32):
    return p + k * (s - e)


def new_elos_after_tournament(p1, p2, s):
    e = 0
    for _p2 in p2:
        e += expected(p1, _p2)
    _p1 = elo(p1, expected(p1, p2), s)
    return _p1


# ---- script
args = ArgumentParser(
    description='run tournament and obtain ELO ratings of different models')
args.add_argument("--m1", type=str, default=".model_sample/z4_0.pt",
                  help="path to first model checkpoint file")
args.add_argument("--m2", type=str, default=".model_sample/z4_0.pt",
                  help="path to second model checkpoint file")
args.add_argument("--num_rounds", type=int, default=800,
                  help="number of rounds in the tournament")
args = args.parse_args()

# make the baseline configuration and load the models
config = ModelConfig(
    vocab_size=1793,  # Fix: Model shape mismatch error
    n_ctx=60,
    n_embd=128,
    n_head=8,
    n_layer=30,
    n_positions=60,
)
m1 = Player(config, args.m1)
m1 = Player(config, args.m2)
