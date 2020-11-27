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

        return self.done, res

################################################
####### Tree ###################################
################################################

def softmax(x, dim= -1):
        n = np.e ** x
        d = np.sum(np.e ** x, axis=dim)
        return (n.T / d).T  # col gets divided instead of row so double transpose

class Node():
    def __init__(self, value, move):
        self.value = value
        self.move = move
        self.children = [] # initialise with a list
        
    @property
    def terminal(self):
        return len(self.childen) > 0
    
    def __eq__(self, n):
        return self.move == n.move
    
    def __len__(self):
        return len(self.children)

    def __repr__(self):
        return f"<Node: move '{self.move[:4]}' ({self.value:.3f})>"

def add_one(model, b, root_node, future_moves, vocab, inv_vocab, flip = True, k = 10):
    # adds one depth of all the predictions to root_node.children
    legal_moves = [vocab[str(x)[:4]] for x in b.legal_moves] # covert to ints for indexing
    mv_batched = [[vocab[str(x)[:4]] for x in b.move_stack] + [l] for l in future_moves] # convert to ints
    mv_batched = torch.Tensor(mv_batched).long()
    logits, val_searched = model(input_ids = mv_batched)
    flip = -1 if True else flip
    for l,v in zip(future_moves, val_searched[:, -1].view(-1).tolist()):
        root_node.children.append(Node(flip * v, inv_vocab[l]))
        
    # now get the top_k moves for all legal moves
    logits = logits[:, -1, legal_moves].detach().numpy() # [N, m]
    log_probs= softmax(logits)  # [N, 1793]
    top_prob_mvids = np.argsort(log_probs, axis = 1)[:, ::-1][:, :k] # [N, k]
    
    return top_prob_mvids, legal_moves


def generate_tree(model, root_node, b, future_moves, vocab, inv_vocab, k = 5):
    """generates a tree by iterating over top policies (breadth) by the model upto
    a specified depth"""

    # do the first search
    top_prob_mvids, legal_moves = add_one(model, b, root_node, future_moves, vocab, inv_vocab, flip = True, k = k)
    top_prob_mvids_strs = [[inv_vocab[legal_moves[i]] for i in tk] for tk in top_prob_mvids] # convert to strings

    for i, child in enumerate(root_node.children):
        # for each child move make a copy of the board add a move and get one depth
        bcopy = b.copy()
        bcopy.push(chess.Move.from_uci(child.move))
        top_prob_mvids_child, lm_child = add_one(model, bcopy, child, future_moves = top_prob_mvids[i], flip = False, k = k)


def minimax(node, depth, maxp, minp, _max = False):
    """
    function minimax(node, depth, maximizingPlayer) is
        if depth = 0 or node is a terminal node then
            return the heuristic value of node
        if maximizingPlayer then
            value := −∞
            for each child of node do
                value := max(value, minimax(child, depth − 1, FALSE))
            return value
        else (* minimizing player *)
            value := +∞
            for each child of node do
                value := min(value, minimax(child, depth − 1, TRUE))
            return value
    """
    if not depth or node.is_terminal:
        return node.value
    
    if _max:
        val = -10000  # −∞
        for child in node:
            val = max(val, minimax(child, depth - 1, maxp, minp, False))
    else:
        val = +10000  # −∞
        for child in node:
            val = min(val, minimax(child, depth - 1, maxp, minp, True))
    return val


################################################
####### Player #################################
################################################

class Player():
    def __init__(self, config, save_path, vocab_path):
        self.tree = None  # callable tree method

        self.config = config
        self.save_path = save_path
        
        with open(vocab_path, "r") as f:
            self.vocab = json.load(f)
            self.inv_vocab = {v: k for k, v in self.vocab.items()}

        # participant code
        self.elo = 1000
        self.idx = None


        self.load()

    def __repr__(self):
        return "<NeuraPlayer>"

    def load(self):
        self.device = "cpu"
        model = BaseHFGPT(self.config)

        # Fixed: Load model in CPU
        model.load_state_dict(torch.load(self.save_path, map_location=torch.device(self.device)))
        model.eval()

        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)
        else:
            self.model = model

    def flush():
        pass

    def better_choice(self, a, p, n):
        # make better choices man!

        # the default np.random.choice has problems when sum(p) != 1
        # which is often the case when normalising using softmax above
        # this code is hacked from the numpy code
        # https://github.com/numpy/numpy/blob/maintenance/1.9.x/numpy/random/mtrand/mtrand.pyx#L1066

        # by default replace = False, we want only unique values
        # and size = 1 only want one sample at a time
        a = np.array(a, copy = False)
        p = p.copy()
        found = np.zeros([n], dtype = np.int)
        m = 0 # how many found
        while m < n:
            x = np.random.rand(n - m)
            if m > 0:
                p[found[0:m]] = 0
            
            cdf = np.cumsum(p)
            cdf /= cdf[-1]
            cdf[-1] = min(cdf[-1], 1.)
            new = cdf.searchsorted(x, side = 'right')
            _, unique_indices = np.unique(new, return_index = True)
            unique_indices.sort()
            new  = new.take(unique_indices)
            found[m:m+new.size] = new
            
            m += new.size

        return a[found]

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
            moves = torch.Tensor(moves).view(1, len(moves)).long().to(self.device)  # [1, N]

            legal = [x for x in b.legal_moves]
            legal_idx = [self.vocab[str(x)[:4]] for x in b.legal_moves]

            # pass to model
            logits, values = model(input_ids=moves)
            logits = logits[0, -1]  # [B,N,V] --> [V]
            values = values[0, -1]  # [B,N,1] --> [1]
            values = values.item()  # scalar

            # softmax over legal moves
            lg_mask = logits.detach().numpy()[legal_idx]
            lg_mask = softmax(lg_mask)
            move = self.better_choice(legal, lg_mask, 1)[0]

            # force promote to queen
            if not str(move)[-1].isdigit():
                move = chess.Move.from_uci(str(move)[:4] + "q")

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
    player2 = Player(config, "models/z5/z5_6000.pt",
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
            mv += 1
        else:
            m, v, c = player2.move(game)
            p = 0
        print(mv, "|", m, v, c)
        done, res = game.step(m)
        pgn_writer_node = pgn_writer_node.add_variation(m)
        if res != "game" or mv == 40:
            print("Game over")
            break
    print(pgn_writer, file=open("latest_game.pgn", "w"), end="\n\n")
