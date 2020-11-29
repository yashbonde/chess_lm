"""
for reference (not used anywhere): https://www.pettingzoo.ml/classic

torch is only used for neural network, the output values are converted to numpy
and all the operations are then used for 
"""
import json
import chess
import chess.pgn
import torch
import numpy as np
from tqdm import trange
from time import time
import torch.nn.functional as F
from model import BaseHFGPT, ModelConfig


def Move(x):
    # get chess.Move object and force promote to queen
    if not x[-1].isdigit():
        x = x[:-1] + "q"
    return chess.Move.from_uci(x)


################################################
####### Engine #################################
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
####### Helper Functions #######################
################################################
def softmax(x, dim=-1):
    n = np.e ** x
    d = np.sum(np.e ** x, axis=dim)
    # col gets divided instead of row so double transpose
    return (n.T / d).T


def top_p(x, p = 0.98):
    # https://arxiv.org/pdf/1904.09751.pdf, using this method reduces the repetitions
    # In practice this means selecting the highest probability tokens whose cumulative probability
    # mass exceeds the pre-chosen threshold p. The size of the sampling set will adjust dynamically
    # based on the shape of the probability distribution at each time step. For high values of p,
    # this is a small subset of vocabulary that takes up vast majority of the probability mass —
    # the nucleus.
    x_sorted = np.sort(x, axis = -1)[:, ::-1]
    x_sorted_idx = np.argsort(x, axis = -1)[:, ::-1]
    cdf = np.cumsum(x_sorted, axis = -1)
    cdf_mask = np.greater(cdf, p)
    
    # need to do list comprehension because there can be a shape mismatch
    cdf_idx = [[x_sorted_idx[i,j] for j,f in enumerate(mask) if f] for i,mask in enumerate(cdf_mask)]
    return cdf_idx


################################################
####### Tree ###################################
################################################
class Node():
    def __init__(self, value, move):
        self.value = value
        self.move = move
        self.children = []  # initialise with a list

    @property
    def terminal(self):
        return len(self.children) == 0

    @property
    def total_nodes(self):
        n = 1
        for c in self.children:
            n += c.total_nodes
        return n

    def __eq__(self, n):
        return self.move == n.move

    def __len__(self):
        return len(self.children)

    def __repr__(self):
        return f"<Move '{self.move[:4]}' ({self.value:.3f}) {len(self)} Child>"

    def __str__(self, level=0):
        ret = "\t"*level+repr(self)+"\n"
        for child in self.children:
            ret += child.__str__(level+1)
        return ret

    @property
    def s(self):
        return self.__repr__()

def one_step(model, b, root, vocab, inv_vocab, mv_ids = None, verbose = False):
    """Take one step into the future and update root's children + return the possible futures for those child"""
    if mv_ids is None:
        # special handling for firstm move
        mv_ids = [vocab[str(x)[:4]] for x in b.legal_moves] # valid moves
    
    if verbose:
        print("Given Root:", root.s, "played on board:", b.fen(), [inv_vocab[x] for x in mv_ids])

    mv_batched = np.asarray([[0] + [vocab[str(x)[:4]] for x in b.move_stack] + [l] for l in mv_ids])[:, :model.config.n_ctx]
    with torch.no_grad():
        logits, values = model(input_ids = torch.Tensor(mv_batched).long())
        logits = logits[:, -1].numpy() # for each move, what is the distribution of future moves
        values = values[:, -1].tolist()

    # now get the possible future possibilities for each move taken
    all_possible_future_moves = []
    if verbose:
        print("\nEntering Future possibilites determiner...")
    for mv, lgt_mv, v_mv in zip(mv_ids, logits, values):
        if verbose:
            print("Applied", inv_vocab[mv], "on board", b.fen(), end = " ")
        # push board to one future and determine what are the legal moves
        bfuture = b.copy()
        bfuture.push(Move(inv_vocab[mv]))
        if verbose:
            print("--> to get board", bfuture.fen())
        future_legal = [vocab[str(x)[:4]] for x in bfuture.legal_moves] # valid futures

        # now get probability distribution for for these legal moves and determine the top ones
        lgt_mv = softmax(lgt_mv[future_legal])
        lgt_mv = top_p(lgt_mv.reshape(1, len(lgt_mv)), p = 0.99)
        future_legal = [future_legal[i] for i in lgt_mv[0]]
        if verbose:
            print("Using Futures", [inv_vocab[x] for x in future_legal], future_legal)
        all_possible_future_moves.append(future_legal)

        # add this child to the root
        root.children.append(Node(v_mv[0], inv_vocab[mv]))
    if verbose:
        print("Completed adding one step future to", root.s, [len(x) for x in all_possible_future_moves], "\n\n")
    return all_possible_future_moves


def generate_tree(model, depth, board, root_node, vocab, inv_vocab, mv_ids = None, verbose = False):
    if verbose:
        print(f"({depth})", root_node.s, "played on", board.fen(), [str(x) for x in board.move_stack[-7:]], "-- FM --", mv_ids)

    # for each child what is the policy and add children to root_node
    all_possible_child_moves = one_step(model = model, b = board, root = root_node, mv_ids=mv_ids,
        verbose = verbose, vocab = vocab, inv_vocab = inv_vocab
    )
    
    if depth > 1:
        if verbose: 
            print("Iterating over Children of", root_node.s)
        for mv_ids, child in zip(all_possible_child_moves, root_node.children):
            # take this child, and make the move on this board
            bchild = board.copy()
            bchild.push(Move(child.move))
            generate_tree(
                model = model, depth = depth - 1, board = bchild, root_node = child, 
                mv_ids=mv_ids, verbose=verbose, vocab=vocab, inv_vocab=inv_vocab
            )


def minimax(node, depth, _max = False):
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
    # print(node.s, _max, depth)
    if not depth or node.terminal:
        return node.value
    
    if _max:
        # print()
        val = -10000  # −∞
        for child in node.children:
            val = max(val, minimax(child, depth - 1, False))
    else:
        # print()
        val = +10000  # −∞
        for child in node.children:
            val = min(val, minimax(child, depth - 1, True))
    return val


################################################
####### Player #################################
################################################

class Player():
    def __init__(self, config, save_path, vocab_path, search = "sample", depth = 1):
        if search not in ["sample", "greedy", "random", "minimax"]:
            raise ValueError(f"Searching method: {search} not defined")

        self.search = search  # method used to determine the move
        self.depth = depth # number of steps to look into the future

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
        # which is often the case when normalising using softmax this code is hacked from the numpy code
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
        # nn predicts the move, value scalar and it's confidence for taking htat move (conf)
        config = self.model.config
        model = self.model
        vocab = self.vocab
        inv_vocab = self.inv_vocab

        # vals to return
        move = None
        value = None
        conf = None

        # define board data to be used with all situations
        b = game.board
        moves = [0] + [self.vocab[str(x)[:4]] for x in b.move_stack] # [0] for game start
        moves = moves[-config.n_ctx:]
        moves = torch.Tensor(moves).view(1, len(moves)).long().to(self.device)  # [1, N]

        legal = [x for x in b.legal_moves]
        legal_idx = [self.vocab[str(x)[:4]] for x in b.legal_moves]

        # pass to model
        with torch.no_grad():
            logits, values = model(input_ids=moves)
            logits = logits[0, -1]  # [B,N,V] --> [V]
            value = values[0, -1].item()  # [B,N,1] --> scalar
            
            # softmax over legal moves, get the log-probabilities for legal moves
            lg_mask = logits.numpy()[legal_idx]
            lg_mask = softmax(lg_mask)

        if self.search == "minimax": # use a minimax method to determine the best possible move
            mv = '[GAME]' if not b.move_stack else str(b.move_stack[-1])[:4] # new game handle
            root_node = Node(value = value, move = mv)

            # generate tree for this node
            _st = time()
            # print("\nStarting tree generation ...", end = " ")
            generate_tree(model=model, depth=self.depth, board=b, root_node=root_node, vocab=vocab, inv_vocab=inv_vocab, verbose = False)
            # print(f"Completed in {time() - _st:.4f}s. {root_node.total_nodes - 1} nodes evaluated")

            # now take the greedy move that maximises the value
            sorted_moves = sorted([
                    (c, minimax(c, 4, _max = True))
                    for c in root_node.children
                ], key = lambda x: x[1]
            ) # Node object
            value = sorted_moves[-1][1]
            move = Move(sorted_moves[-1][0].move)

        elif self.search == "sample":
            # no searching, make a choice based on probability distribution
            move = self.better_choice(legal, lg_mask, 1)[0]
            conf = np.max(lg_mask)

        elif self.search == "greedy":
            # greedy method
            move = legal[np.argmax(lg_mask)]

        elif self.search == "random":
            # random moves
            move = np.random.choice(legal)
            value = None
            conf = None

        return move, value, conf


# test script
if __name__ == "__main__":
    with open("assets/moves.json") as f:
        vocab_size = len(json.load(f))
    config = ModelConfig(vocab_size=vocab_size, n_positions=60,
                         n_ctx=60, n_embd=128, n_layer=30, n_head=8)
    player1 = Player(config, "models/z4/z4_0.pt",
                     "assets/moves.json", search="sample")  # assume to be black
    config = ModelConfig(vocab_size=vocab_size, n_positions=180,
                         n_ctx=180, n_embd=128, n_layer=30, n_head=8)
    player2 = Player(config, "models/q1/q1_15000.pt",
                     "assets/moves.json", search="minimax",
                     depth = 1)  # assume to be white

    for round in trange(100):
        print(f"Starting round: {round}")
        game = GameEngine()
        pgn_writer = chess.pgn.Game()
        pgn_writer_node = pgn_writer
        pgn_writer.headers["Event"] = "Test with p=0.99 vs 0.999 in prev"
        pgn_writer.headers["White"] = "z4_0"
        pgn_writer.headers["Black"] = "q1_15000"
        pgn_writer.headers["Round"] = str(round)
        
        # play
        mv = 0
        done = False
        p = 0
        try:
            while not done:
                # model returns move object, value, confidence of move
                if p == 0:
                    m, v, c = player1.move(game)
                    p += 1
                    mv += 1
                else:
                    m, v, c = player2.move(game)
                    p = 0
                # print(mv, "|", m, v, c)
                done, res = game.step(m)
                pgn_writer_node = pgn_writer_node.add_variation(m)
                if res != "game" or mv == 50:
                    print("Game over")
                    break
        except KeyboardInterrupt:
            break
        print(pgn_writer, file=open("auto_tournaments_search_d1.pgn", "a"), end="\n\n")
