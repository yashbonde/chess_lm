"""
for reference (not used anywhere): https://www.pettingzoo.ml/classic
"""
import json
import random
import chess
import chess.pgn
import torch
import numpy as np
import torch.nn.functional as F
from model import BaseHFGPT, ModelConfig

####### Engine #################################
# Here I am writing a simple game engine that is NOT
# borrowed from my another project that is a complete
# Full Stack chess app, insted this is a much more
# functional and generic. Most likely I will port this
# over to my other repo
################################################


class GameEngine():
    def __init__(self):
        self.board = chess.Board()  # initialise an empty board
        self.done = False

    def __repr__(self):
        return str(self.board)

    @property
    def fen(self):
        return self.board.fen()

    @property
    def legal_moves(self):
        return self.board.legal_moves

    def reset(self):
        self.board.reset()

    def step(self, move_id):
        """
        Game is considered draw in the following cases:
        - Stalemate: no legal move but no check
        - Threefold repetition rule: an identical position has occured in last three moves
            (player has to ask the arbiter to declare draw or mutually agree)
        - Fifty-move rule: if in the previous 50 moves by ecah side, no pawn has moved and no
            capture has been made (player has to ask the arbiter to declare draw or mutually agree)
        - Fivefold repetition: an identical position has occured in last five moves (auto-draw)
        - Seventyfive-move rule: if in the previous 75 moves by ecah side, no pawn has moved and no
            capture has been made (auto-draw)
        - Insufficient Material: this includes:
            - K vs k
            - K & B vs k
            - K & N vs k
            - K & B vs k & b (Bb on same colour)
        - Mutual Agreement

        Read more: https://en.wikipedia.org/wiki/Glossary_of_chess
        Read more: https://www.chess.com/article/view/how-chess-games-can-end-8-ways-explained
        """
        board = self.board
        board.push(move_id)
        res = "game"

        # draw results
        if board.is_stalemate():  # stalemate
            print("Stalemate")
            self.done = True
            res = "draw"

        elif board.can_claim_threefold_repetition():
            print("Threefold repetition claimed")
            self.done = True
            res = "draw"

        elif board.can_claim_fifty_moves():
            print("Fifty Moves claimed")
            self.done = True
            res = "draw"

        elif board.is_fivefold_repetition():
            print("Fivefold repetition")
            self.done = True
            res = "draw"

        elif board.is_seventyfive_moves():
            print("SeventyFive Moves")
            self.done = True
            res = "draw"

        elif board.is_insufficient_material():
            print("Insufficient Material")
            self.done = True
            res = "draw"

        # win result
        elif board.is_checkmate():
            print("Checkmate")
            self.done = True
            res = "win"

        # this is generic and does not tell the result
        # elif board.is_game_over():
        #     print("Game Over")
        #     self.done = True
        #     res = "draw"
        return self.done, res

################################################
####### Player #################################
################################################


class Player():
    def __init__(self, config, save_path, vocab_path):
        model = BaseHFGPT(config)
        self.device = "cpu"
        self.tree = None  # callable tree method

        # Fixed: Load model in CPU
        model.load_state_dict(torch.load(
            save_path, map_location=torch.device(self.device)))
        model.eval()

        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)
        else:
            self.model = model

        with open(vocab_path, "r") as f:
            self.vocab = json.load(f)
            self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def __repr__(self):
        return "<NeuraPlayer>"

    def softmax(self, x):
        x = (np.e ** x) / sum(np.e ** x)
        # s = np.sum(x)
        # print("\n", s - 1)
        # if s - 1 < 0:
        #     x[0] += 1 - s
        # elif s - 1 > 0:
        #     x[0] -= 1 - s
        return x

    def move(self, game):
        # nn predicts the move and it's confidence on outcome (value?)
        config = self.model.config
        model = self.model

        if self.tree is not None:
            # self.tree: a tree callable object that takes in the model and board state
            move = self.tree(model, game)

        else:
            b = game.board
            # this is no search algorithm / greedy sampled search
            moves = [0] + [self.vocab[str(x)[:4]] for x in b.move_stack]
            moves = moves[:config.n_ctx]
            moves = torch.Tensor(moves).view(
                1, len(moves)).long().to(self.device)  # [1, N]

            # print(moves)

            legal = [x for x in b.legal_moves]
            legal_idx = [self.vocab[str(x)[:4]] for x in b.legal_moves]
            # legal_mask = np.ones(config.vocab_size, dtype=np.float32) * 1e-6
            # legal_mask[legal_idx] = 0
            # legal_mask = torch.Tensor(legal_mask).to(self.device)  # [V]

            # print(legal_mask, sum(legal_mask))

            # pass to model
            logits, values = model(input_ids=moves)
            logits = logits[0, -1]  # [B,N,V] --> [V]
            values = values[0, -1]  # [B,N,1] --> [1]
            values = values.item()  # scalar

            # softmax over legal moves
            lg_mask = logits.detach().numpy()[legal_idx]
            lg_mask = self.softmax(lg_mask)
            # print(lg_mask)

            idx = np.random.multinomial(1, lg_mask,) # multinomial can take probabs that almost
            # print(lg_mask.tolist(), np.argmax(idx))
            # print(np.argmax(idx))

            # if np.sum(lg_mask) != 1:
            #     print("MMMM")
            #     lg_mask /= np.sum(lg_mask)

            # print(np.sum(lg_mask), np.sum(lg_mask) == 1)

            # lg_mask = F.softmax(logits + legal_mask)

            # if self.device != "cpu":
            #     lg_mask = lg_mask.cpu()
            # lg_mask = lg_mask.detach().numpy().astype(np.float32)[legal_idx]


            # add code for sampling moves
            # lg_mask = lg_mask/lg_mask.sum(axis=0, keepdims=1)
            # lg_mask /= lg_mask.sum()
            # move = np.random.choice(legal, size = 1, p = lg_mask)
            # print(lg_mask, sum(lg_mask))

            # move = legal[np.argmax(lg_mask)]
            move = legal[np.argmax(idx)]
            return move, values, np.max(lg_mask)

    def make_random_move(self, b):
        legal_moves = list(b.legal_moves)
        if not legal_moves:
            return None
        move = random.choice(legal_moves)
        return move


# define a random player for prototyping
class RandomPlayer():
    def __init__(self, id):
        self.id = id

    def move(self, b):
        legal_moves = list(b.legal_moves)
        if not legal_moves:
            return None
        move = random.choice(legal_moves)
        return move

    def __repr__(self):
        return f"<Random Player {self.id}>"


# test script
if __name__ == "__main__":

    with open("assets/moves.json") as f:
        vocab_size = len(json.load(f))
    config = ModelConfig(vocab_size=vocab_size, n_positions=60,
                         n_ctx=60, n_embd=128, n_layer=30, n_head=8)

    game = GameEngine()
    pgn_writer = chess.pgn.Game()
    pgn_writer_node = pgn_writer
    pgn_writer.headers["Event"] = "Test"

    player1 = Player(config, ".model_sample/z4_0.pt",
                     "assets/moves.json")  # assume to be white
    player2 = Player(config, ".model_sample/z4_0.pt",
                     "assets/moves.json")  # assume to be black

    # play
    mv = 0
    done = False
    p = 0
    while not done:
        # model returns move object, value, confidence of move
        if p == 0:
            m, v, c = player1.move(game)
            p += 1
        else:
            m, v, c = player2.move(game)
            p = 0
        done, res = game.step(m)
        pgn_writer_node = pgn_writer_node.add_variation(m)

        if res != "game":
            print("Game over")
            break

        print(m, v, c)
    print(pgn_writer, file=open("latest_game.pgn", "w"), end="\n\n")
