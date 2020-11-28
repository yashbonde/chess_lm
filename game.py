"""
for reference (not used anywhere): https://www.pettingzoo.ml/classic

torch is only used for neural network, the output values are converted to numpy
and all the operations are then used for 
"""
import json
import random
import chess
import chess.pgn
import torch
import numpy as np
import torch.nn.functional as F
from model import BaseHFGPT, ModelConfig

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
        self.children = [] # initialise with a list

    @property
    def terminal(self):
        return len(self.children) == 0

    def __eq__(self, n):
        return self.move == n.move

    def __len__(self):
        return len(self.children)

    def __repr__(self):
        return f"<Node: move '{self.move[:4]}' ({self.value:.3f})>"

    def __str__(self, level=0):
        ret = "\t"*level+repr(self)+"\n"
        for child in self.children:
            ret += child.__str__(level+1)
        return ret

    @property
    def s(self):
        return f"<Node: move '{self.move[:4]}' ({self.value:.3f})>"


def add_one(model, b, root_node, future_moves, vocab, inv_vocab, flip = True, k = 10):
    # adds one depth of all the predictions to root_node.children
    legal_moves = [vocab[str(x)[:4]] for x in b.legal_moves] # covert to ints for indexing
    mv_batched = [[vocab[str(x)[:4]] for x in b.move_stack] + [l] for l in future_moves] # convert to ints
    mv_batched = torch.Tensor(mv_batched).long()
    with torch.no_grad():
        # no need to to do .detach() now
        logits, val_searched = model(input_ids = mv_batched)
    flip = -1 if True else flip
    for l,v in zip(future_moves, val_searched[:, -1].view(-1).tolist()):
        root_node.children.append(Node(flip * v, inv_vocab[l]))
        
    # now get the top-probab legal moves
    logits = logits[:, -1, legal_moves].numpy() # [N, m]
    probs = softmax(logits)  # [N, 1793]

    # top-k This is not as good as top-p
    # top_prob_mv_idx = np.argsort(log_probs, axis = 1)[:, ::-1][:, :k] # [N, k]
    # top_prob_mvids = [[legal_moves[i] for i in tk] for tk in top_prob_mv_idx]

    # top-p
    top_prob_mv_idx = top_p(probs, 0.9999)
    top_prob_mvids = [[legal_moves[i] for i in tk] for tk in top_prob_mv_idx]
    return top_prob_mvids


def generate_tree(model, depth, root_node, b, future_moves, vocab, inv_vocab, k = 5):
    """generates a tree by iterating over top policies (breadth) by the model upto a specified depth"""
    print(f"\n ({depth}) Already played:", root_node.s, "giving -->", b.fen(), "Possible Moves:", [inv_vocab[x] for x in future_moves])
    if depth == 0:
        return

    # do the first search
    top_prob_mvids = add_one(model = model, b = b, root_node = root_node, future_moves = future_moves, vocab = vocab, inv_vocab = inv_vocab, flip = False, k = k)
    top_prob_mvids_strs = [[inv_vocab[i] for i in tk] for tk in top_prob_mvids] # convert to strings
    print("top_prob_mvids_strs:", top_prob_mvids_strs, top_prob_mvids)
    print(root_node)

    for i, child in enumerate(root_node.children):
        # for each child move make a copy of the board add a move and get one depth
        print("Playing:", child.s, "on", b.fen())
        bcopy = b.copy()
        bcopy.push(chess.Move.from_uci(child.move))
        generate_tree(model = model, depth = depth - 1, root_node = child, b = bcopy, future_moves = top_prob_mvids[i], vocab = vocab, inv_vocab = inv_vocab, k = k)

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
    def __init__(self, config, save_path, vocab_path, search = "sample"):
        if search not in ["sample", "random", "minimax"]:
            raise ValueError(f"Searching method: {search} not defined")

        self.search = search  # method used to determine the move

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
        moves = moves[:config.n_ctx]
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


            # define the possible future moves for this I am using nucleus sampling or top_p sampling
            # https://arxiv.org/pdf/1904.09751.pdf, using this method reduces the repetitions
            # In practice this means selecting the highest probability tokens whose cumulative probability
            # mass exceeds the pre-chosen threshold p. The size of the sampling set will adjust dynamically
            # based on the shape of the probability distribution at each time step. For high values of p,
            # this is a small subset of vocabulary that takes up vast majority of the probability mass —
            # the nucleus.
            cumulative_probability_mass = np.cumsum(np.sort(lg_mask)[::-1])
            future_moves = [legal_idx[i] for i,t in enumerate(cumulative_probability_mass > 0.98) if t]

            # define a root node and create a tree for that root node (add to node.children)
            root_node = Node(str(b.move_stack[-1]), ) # why call it stack it's a queue
            generate_tree(
                model=model,
                depth=4,
                root_node=root_node,
                b=game.board,
                future_moves=future_moves,
                vocab=vocab,
                inv_vocab=inv_vocab,
                k=5
            )
            move = np.random.choice(legal)
            print(root_node)

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

        # force promote to queen
        if not str(move)[-1].isdigit():
            move = chess.Move.from_uci(str(move)[:4] + "q")

        return move, value, conf


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
        if res != "game" or mv == 10:
            print("Game over")
            break
    print(pgn_writer, file=open("latest_game.pgn", "w"), end="\n\n")
