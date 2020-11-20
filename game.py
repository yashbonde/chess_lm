
import random
import chess
import torch
from model import BaseHFGPT

################################################
####### Engine #################################
# Here I am writing a simple game engine that is borrowed from my another
# project that is a complete Full Stack chess app
################################################


class GameEngine():
    def __init__(self):
        self.board = chess.Board()  # initialise an empty board

    def __repr__(self):
        return str(self.board)

    def step(self, move_id):
        board = self.board
        if board.is_checkmate():
            return "checkmate", board.fen(), None
        elif board.is_stalemate():
            return "stalemate", board.fen(), None


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

    # def take_step(self, moves):
    #     obs = np.asarray(moves).view(1, len(moves))


class RandomPlayer():
    def __init__(self):
        pass

    def move(self, b):
        legal_moves = list(b.legal_moves)
        if not legal_moves:
            return None
        move = random.choice(legal_moves)
        return move

    def __repr__(self)
    



# test script

if __name__ == "__main__":
    game = GameEngine()
    player1 = RandomPlayer()  # assume to be white
    player1 = RandomPlayer()  # assume to be black

    print(game)

