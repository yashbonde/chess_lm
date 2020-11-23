"""
for reference (not used anywhere): https://www.pettingzoo.ml/classic
"""
import random
import chess
import torch
import torch.nn.functional as F
from model import BaseHFGPT

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
        """
        board = self.board
        board.push(move_id)
        res = "game"

        # draw results
        if board.is_stalemate(): # stalemate
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
    def __init__(self, config, save_path, mcts=False):
        model = BaseHFGPT(config)
        model.load_state_dict(torch.load(save_path))
        model.eval()

        self.mcts = mcts  # to use mcts or not

        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def __repr__(self):
        return "<Player>"

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
    game = GameEngine()
    player1 = RandomPlayer(1)  # assume to be white
    player2 = RandomPlayer(2)  # assume to be black

    print(player1, player2)
    print(game)

    mv = 0
    done = False
    p = 0
    while not done:
        if p == 0:
            m = player1.move(game.board)
            p += 1
        else:
            m = player2.move(game.board)
            p = 0
        done, res = game.step(m)

        if res == "win":
            p = f"Player{p+1} wins"
        elif res == "draw":
            p = f"{p} game draw"

        print(f"==== {p} ====")
        print(game.fen)

        mv += 1
        if mv == 1000:
            print("Thousand Line break")
            break


