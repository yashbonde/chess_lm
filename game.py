"""
for reference (not used anywhere): https://www.pettingzoo.ml/classic

torch is only used for neural network, the output values are converted to numpy
and all the operations are then used for
"""
import json
import chess
import chess.pgn
import numpy as np
from time import time
from tqdm import trange
from types import SimpleNamespace

import torch
from torch import Tensor
import torch.nn.functional as F
from model import BaseHFGPT, ModelConfig, BetaChess


################################################
####### Helper Functions #######################
################################################

def get_model_config(model):
    return model.module.config if hasattr(model, "module") else model.config

def verbose_print(*args, verbose):
    # print only when verbose is True
    if verbose:
        print(*args)


def Move(x):
    # get chess.Move object and force promote to queen
    if not x[-1].isdigit():
        x = x[:-1] + "q"
    return chess.Move.from_uci(x)


def get_mv_ids(b, vocab):
    # return move ids for this game's move stack
    return [vocab["[GAME]"]] + [vocab[str(x)[:4]] for x in b.move_stack]


def check_board_state(b, verbose = False):
    # check the board's state and return appropriate response
    done = False
    res = "game"
    # draw results
    if b.is_stalemate():  # stalemate
        verbose_print("Stalemate", verbose = verbose)
        done = True
        res = "draw"

    elif b.can_claim_threefold_repetition():
        verbose_print("Threefold repetition claimed", verbose = verbose)
        done = True
        res = "draw"

    elif b.can_claim_fifty_moves():
        verbose_print("Fifty Moves claimed", verbose = verbose)
        done = True
        res = "draw"

    elif b.is_fivefold_repetition():
        verbose_print("Fivefold repetition", verbose = verbose)
        done = True
        res = "draw"

    elif b.is_seventyfive_moves():
        verbose_print("SeventyFive Moves", verbose = verbose)
        done = True
        res = "draw"

    elif b.is_insufficient_material():
        verbose_print("Insufficient Material", verbose = verbose)
        done = True
        res = "draw"

    # win result
    elif b.is_checkmate():
        verbose_print("Checkmate", verbose = verbose)
        done = True
        res = "win"
    return done, res


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

    def step(self, move, verbose = False):
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
        board.push(move)
        res = "game"
        done, res = check_board_state(self.board, verbose = verbose)
        self.done = done
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
####### Tree Helpers ###########################
################################################
class Node():
    def __init__(self, state_value, move, p, b, color, rc, is_root = False):
        self.state_value = state_value # the value is now state action value
        self.move = move
        self.p = p # prior - probability of taking this move
        self.color = color # color of player taking this move
        self.rc = rc # root_node.color
        self.is_root = is_root # root node flag used for action selection

        self.n = 0 # number of times this is visited, initialised with 0
        self.children = [] # all the children
        self.state = [str(x) for x in b.move_stack] # state at which this action was taken
        self.nsb = len(b.move_stack)
        self.action_value = 0 # what is the action value of taking this move
        self.inv_color = "white" if color == "black" else "black"

    @property
    def total_nodes(self):
        n = 1
        for c in self.children:
            n += c.total_nodes
        return n

    def all_children(self):
        c = []
        if self.children:
            for child in self.children:
                c.extend(child.all_children())
        else:
            c = [self]
        return c

    def get_adjusted_action_value(self):
        # this is the one given in MuZero paper
        const = 1.25 + np.log(1 + (1 + self.nsb)/19652) * (np.sqrt(self.nsb) / (1+self.n))
        value = self.action_value + self.p * const
        return value

    def get_action_value_with_exploration(self):
        return self.action_value + 5 * self.p * np.sqrt(self.nsb) / (1+self.n)

    @property
    def terminal(self):
        return len(self.children) == 0

    def __eq__(self, n):
        return self.move == n.move

    def __len__(self):
        return len(self.children)

    def __repr__(self):
        return f"<Move '{self.move}'; q={self.action_value:.3f} v={self.state_value:.3f} c={len(self)}; n={self.n}; s={len(self.state)} nsb={self.nsb} p={self.p:.3f}>"

    def __str__(self, level=0):
        ret = "  "*level+repr(self)+"\n"
        for child in self.children:
            ret += child.__str__(level+1)
        return ret

    @property
    def s(self):
        return self.__repr__()


def one_future(mv, mvp, lgt_mv, v_mv, vocab, inv_vocab, b, p = 0.98, verbose = False):
    verbose_print("Applied", inv_vocab[mv], "on board", b.fen(), end=" ", verbose = verbose)
    # push board to one future and determine what are the legal moves
    bfuture = b.copy()
    bfuture.push(Move(inv_vocab[mv]))
    if verbose:
        verbose_print("--> to get board", bfuture.fen(), verbose = verbose)
    future_legal = [vocab[str(x)[:4]] for x in bfuture.legal_moves] # valid futures

    # now get probability distribution for for these legal moves and determine the top ones
    lgt_mv = softmax(lgt_mv[future_legal])
    lgt_mv_idx = top_p(lgt_mv.reshape(1, len(lgt_mv)), p=p)  # [1, N]
    future_legal = [future_legal[i] for i in lgt_mv_idx[0]]
    lgt_mv_probs = [lgt_mv[i] for i in lgt_mv_idx[0]]
    verbose_print("Using Futures", [inv_vocab[x] for x in future_legal], future_legal, verbose = verbose)
    return future_legal, lgt_mv_probs, Node(value=v_mv[0], move=inv_vocab[mv], p=mvp, b=bfuture)


def one_step(model, b, root, vocab, inv_vocab, mv_ids = None, mv_probs = None, verbose = False, p = 0.98):
    """Take one step into the future and update root's children + return the possible futures for those child"""
    if mv_ids is not None:
        assert mv_probs is not None, "Provide probability as well"
        assert len(mv_ids) == len(mv_probs)

    if mv_ids is None:
        # special handling for firstm move
        mv_ids = [vocab[str(x)[:4]] for x in b.legal_moves] # valid moves
        mv_probs = np.ones(len(mv_ids), dtype=np.float) / len(mv_ids)

    verbose_print("Given Root:", root.s, "played on board:", b.fen(), [inv_vocab[x] for x in mv_ids], verbose = verbose)

    mv_batched = np.asarray([[0] + [vocab[str(x)[:4]] for x in b.move_stack] + [l] for l in mv_ids])[:, :get_model_config(model).n_ctx]
    with torch.no_grad():
        logits, values = model(input_ids = torch.Tensor(mv_batched).long())
        logits = logits[:, -1].numpy() # for each move, what is the distribution of future moves
        values = values[:, -1].tolist()

    # now get the possible future possibilities for each move taken
    all_possible_future_moves = []
    all_possible_future_moves_probs = []
    verbose_print("\nEntering Future possibilites determiner...", verbose = verbose)

    for mv, mvp, lgt_mv, v_mv in zip(mv_ids, mv_probs, logits, values):
        # make one step in the future and return the best value
        future_legal, lgt_mv_probs, node = one_future(mv, mvp, lgt_mv, v_mv, vocab, inv_vocab, b, p =p, verbose =verbose)
        all_possible_future_moves.append(future_legal)
        all_possible_future_moves_probs.append(lgt_mv_probs)
        root.children.append(node)
    verbose_print("Completed adding one step future to", root.s, [len(x) for x in all_possible_future_moves], "\n\n", verbose = verbose)
    return all_possible_future_moves, all_possible_future_moves_probs


def generate_tree(model, depth, board, root_node, vocab, inv_vocab, mv_ids = None, mv_probs = None, verbose = False):
    verbose_print(f"({depth})", root_node.s, "played on", board.fen(), [str(x) for x in board.move_stack[-7:]], "-- FM --", mv_ids, verbose = verbose)

    # for each child what is the policy and add children to root_node
    all_possible_future_moves, all_possible_future_moves_probs = one_step(model=model, b=board, root=root_node, mv_ids=mv_ids,
                                        verbose=verbose, vocab=vocab, inv_vocab=inv_vocab, mv_probs=mv_probs
    )

    if depth > 1:
        verbose_print("Iterating over Children of", root_node.s, verbose = verbose)
        for mv_ids, mv_probs, child in zip(all_possible_future_moves, all_possible_future_moves_probs, root_node.children):
            # take this child, and make the move on this board
            bchild = board.copy()
            bchild.push(Move(child.move))
            generate_tree(
                model = model, depth = depth - 1, board = bchild, root_node = child,
                mv_ids=mv_ids, verbose=verbose, vocab=vocab, inv_vocab=inv_vocab,
                mv_probs=mv_probs
            )


def update_node_bonus(root_node):
    # update the `n` in all the nodes
    ac = root_node.all_children()
    unq = []
    for c in ac:
        if c not in unq:
            unq.append(c)

    # create a counter --> horrible code
    cntr = {}
    for c in unq:
        for cc in ac:
            if c == cc:
                if c.move not in cntr:
                    cntr[c.move] = 1
                else:
                    cntr[c.move] += 1

    # now go over the children and update those
    for c in ac:
        c.n = cntr[c.move]

    return root_node


def minimax(node, depth, _max=False):
    # verbose_print(node.s, _max, depth, verbose = verbose)
    if not depth or node.terminal:
        return node.state_value

    if _max:
        # verbose_print(, verbose = verbose)
        val = -10000  # −∞
        for child in node.children:
            val = max(val, minimax(child, depth - 1, False))
    else:
        # verbose_print(, verbose = verbose)
        val = +10000  # −∞
        for child in node.children:
            val = min(val, minimax(child, depth - 1, True))
    return val

################################################
####### Monte Carlo Tree Search ################
################################################
def expand_tree(model, root_node, b, depth, vocab, inv_vocab, nodes_taken):
    mvs = get_mv_ids(b, vocab)[:get_model_config(model).n_ctx] # whatever has been played till now
    legal_moves = [vocab[str(x)[:4]] for x in b.legal_moves]
    _CUDA = "cuda" in str(next(model.parameters()).device)

    # step 1: for this root node I need to determine what are the probabilities of next move
    with torch.no_grad():
        # before we were directly using the CPU version now we perform softmax on GPU
        # logits_kp1 = logits[0, -1].numpy()  # [1, N]
        # logits_kp1 = softmax(logits_kp1[legal_moves])
        if _CUDA:
            logits, values = model(input_ids=Tensor(mvs).view(1, len(mvs)).long().to("cuda:0"))
            logits = F.softmax(logits[0, -1, legal_moves], dim = -1).cpu().numpy()
        else:
            logits, values = model(input_ids=Tensor(mvs).view(1, len(mvs)).long())
            logits = softmax(logits[0, -1, legal_moves].numpy())

    # EXPANSION
    # step 2: for all the next legal moves what are the probabilities and next board states
    mvs_batched = np.asarray([[*mvs] + [l] for l in legal_moves])[:, :get_model_config(model).n_ctx]
    with torch.no_grad():
        # logits_kp2 = logits[0, -1].numpy()  # [1, N] # at k+2
        if _CUDA:
            logits, values = model(input_ids=Tensor(mvs_batched).long().to("cuda:0"))
            logits_kp1 = F.softmax(logits[0, -1, legal_moves], dim = -1).cpu().numpy()
            values_kp2 = values[:, -1].cpu().tolist()  # [1, 1]
        else:
            logits, values = model(input_ids=Tensor(mvs_batched).long())
            logits_kp1 = softmax(logits[0, -1, legal_moves].numpy())
            values_kp2 = values[:, -1].tolist()
    # now for this root_node we have the k+2 depth possible tree, so we add all this to the root_node children
    for prior, state_value_kp2, mv in zip(logits_kp1, values_kp2, legal_moves):
        # the priors are for k+1 and so come from logits_kp1 while the state values come from k+2
        node = Node(
            state_value = state_value_kp2[0],
            move = inv_vocab[mv],
            p = prior,
            b = b,
            color=root_node.inv_color,
            rc=root_node.rc
        )
        this_node = list(filter(lambda x: x == node, root_node.children))
        if this_node:
            this_node = this_node[0]
            this_node.p = node.p
        else:
            root_node.children.append(node)

    # SELECTION
    # step 3: select the best move
    action_values = np.asarray([x.get_action_value_with_exploration() for x in root_node.children]) # a = q + u
    if root_node.is_root:
        # From Paper: Dirichlet noise Dir(α) was added to the prior probabilities in the root node;
        # this was scaled in inverse proportion to the approximate number of legal moves in a typical position
        dirchlet_noise =  np.random.dirichlet(np.ones_like(action_values) * 0.3) # Dir(a)
        # dirchlet_noise = dirchlet_noise ** (1/len(legal_moves))
        action_values = 0.75 * action_values + 0.25 * dirchlet_noise
    child_node = root_node.children[np.argsort(action_values)[-1]] # chose the action with highest state action value
    nodes_taken.append(child_node)
    # now we have the best move so push this to the board
    bfuture = b.copy()
    bfuture.push(Move(child_node.move))

    # EVALUATION
    # step 4: check for terminal state
    done, res = check_board_state(bfuture)
    if done:
        if res == "draw":
            child_node.state_value = 0
        else:
            value = 1 if child_node.color == child_node.rc else -1
            child_node.state_value = value
    else:
        if depth > 1:
            # expansion
            expand_tree(
                model=model,
                root_node=child_node,
                b=bfuture,
                depth=depth - 1,
                vocab=vocab,
                inv_vocab=inv_vocab,
                nodes_taken=nodes_taken
            )


def backup(nodes_taken, discount_factor = 1):
    # BACKUP
    # discounted bootstrap backup for action_value updation. undiscounted means gamma = 1
    for i, n in enumerate(nodes_taken[:-1]):
        bootstrap = 0
        for j, n2 in enumerate(nodes_taken[i+1:]):
            bootstrap += (discount_factor**j) * n2.state_value
        n.action_value = (n.n*n.action_value + bootstrap) / (n.n + 1)
        n.n = n.n + 1

def mcts(model, root_node, b, depth, vocab, inv_vocab, sims = 10, _trange = True):
    pbar = trange(sims) if _trange else range(sims)
    for i in pbar:
        nodes_taken = []
        expand_tree(model, root_node, b, depth, vocab, inv_vocab, nodes_taken)
        backup(nodes_taken)

def print_mcts(root_node, level=0):
    str_level = str(level) + " "*(3 - len(str(level)))
    ret = str_level + "  "*level+repr(root_node)+"\n"
    for child in root_node.children:
        if len(child):
            ret += print_mcts(child, level + 1)
    return ret


def select_action(n, t=1):
    counts = np.array([x.n for x in n.children])
    policy = (counts ** (1/t)) / sum(counts ** (1/t))
    return policy


################################################
####### Self-Play ##############################
################################################
def self_play_one_game(
        m1, m2, vocab, inv_vocab, game_id, win_col, replay_buffer = None,
        max_moves = 10, depth = 10, sims = 10,
        verbose = False, _trange_moves = True, _trange = False,
        _CUDA = False
    ):
    """
    plays one game between two players m1 and m2, assumes m1 = white and m2 = black
    m1: Player object
    m2: Player object
    vocab: vocab
    inv_vocab: inv_vocab
    game_id: this game ID
    replay_buffer: list in which this games data will be extended
    max_moves: maximum numebr of moves to take in the game
    depth: depth of the search tree
    sims: number of simulations to do for MCTS
    """
    # we are going to start the game from scratch
    game = GameEngine()
    col = None
    res = None
    this_game_buffer = []
    pbar = trange(max_moves) if _trange_moves else range(max_moves)
    for mid in pbar:
        # get player color, first step is always white and then rest of the moves are alternate
        col = "white" if (mid == 0) or ((mid + 1) % 2 == 0) else "black"
        model = m1 if col == "white" else m2 # get the player model
        b = game.board
        verbose_print(b.fen(), verbose = verbose)

        # now board is player till n_moves, root_node (k=0, s^0) = s_{n_moves}
        legal_moves = [vocab[str(x)[:4]] for x in b.legal_moves]
        moves = get_mv_ids(b, vocab)
        with torch.no_grad():
            if _CUDA:
                logits, value = model(input_ids = Tensor(moves).view(1, len(moves)).long().to("cuda:0"))
                logits = F.softmax(logits[0, -1, legal_moves], dim = -1).cpu().numpy()
            else:
                logits, value = model(input_ids = Tensor(moves).view(1, len(moves)).long())
                logits = softmax(logits[0, -1, legal_moves].numpy())
            value = value[0, -1].item()
        # root_node = Node(state_value = values, move = "[GAME]", p = 0., b = b, color = "white", rc = "white")
        move_str = str(b.move_stack[-1]) if b.move_stack else "[GAME]"
        root_node = Node(
            state_value = value,
            move = move_str,
            p = 0.,
            b = b,
            color = "white",
            rc = "white",
            is_root = True
        )

        # update buffer with the last move taken
        this_game_buffer.append(SimpleNamespace(
            move_id = vocab[move_str],
            value = value,
            game_id = game_id
        ))

        # perform mcts and get the policy distribution
        mcts(model, root_node, b, depth, vocab, inv_vocab, sims = sims, _trange = _trange)
        policy = select_action(root_node, t = 1) # larger temp, more variance
        action = Player.better_choice(legal_moves, policy, n = 1)[0]
        move = Move(inv_vocab[action])
        verbose_print(policy, action, "--->", move, verbose = verbose)
        done, res = game.step(move)
        end_value = 0.
        if done:
            verbose_print(f"Game is over at step {mid + 1} and player color {col} --> {res}", verbose = verbose)
            if res == "win" and win_col == col: # when winning color is same as learning model
                end_value = +1.
            elif res == "win" and win_col != col: # when winning color is opp to learning model
                end_value = -1.
            break

    # add the last move taken in the buffer
    this_game_buffer.append(SimpleNamespace(
        move_id = vocab[str(move)],
        value = end_value,
        game_id=game_id
    ))
    verbose_print("-"*10, end_value, verbose = verbose)

    if replay_buffer is not None:
        replay_buffer.extend(this_game_buffer)
        return col, res, replay_buffer
    else:
        return col, res, this_game_buffer

################################################
####### Player #################################
################################################
class Player():
    def __init__(self, config, save_path, vocab_path, model_class, search = "sample", depth = 1):
        if search not in ["sample", "greedy", "minimax", "mcts"]:
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

        self.load(model_class)

    def __repr__(self):
        return "<NeuraPlayer>"

    def load(self, model_class):
        self.device = "cpu"
        model = model_class(self.config)

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

    @staticmethod
    def better_choice(a, p, n):
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

    def move(self, game, verbose = False):
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
        mv = '[GAME]' if not b.move_stack else str(b.move_stack[-1])[:4] # new game handle

        if self.search == "minimax": # use a minimax method to determine the best possible move
            root_node = Node(state_value = value, move = mv, p = 0., b = b)
            # generate tree for this node
            _st = time()
            # verbose_print("\nStarting tree generation ...", end = " ", verbose = verbose)
            generate_tree(
                model=model,
                depth=self.depth,
                board=b,
                root_node=root_node,
                vocab=vocab,
                inv_vocab=inv_vocab,
                verbose = False,
            )
            root_node.update_q()
            # verbose_print(f"Completed in {time() - _st:.4f}s. {root_node.total_nodes - 1} nodes evaluated", verbose = verbose)
            # root_node = update_node_bonus(root_node)

            # now take the greedy move that maximises the value
            sorted_moves = sorted([
                    (c, minimax(c, 4, _max = True))
                    for c in root_node.children
                ], key = lambda x: x[1]
            ) # Node object
            move_node = sorted_moves[-1][0]
            value = move_node.state_value
            move = Move(move_node.move)
            conf = move_node.p

        elif self.search == "mcts":
            col = b.fen().split()[1]
            col = "black" if col == "b" else "white"
            root_node = Node(state_value=value, move=mv, p=0., b=b, color=col, rc=col)
            mcts(
                model=model,
                root_node=root_node,
                b=b,
                depth=10,
                vocab=vocab,
                inv_vocab=inv_vocab,
                sims=100
            )
            move_probab = select_action(root_node, 0.65)
            move = self.better_choice(legal, move_probab, 1)[0]

        elif self.search == "sample":
            # no searching, make a choice based on probability distribution
            move = self.better_choice(legal, lg_mask, 1)[0]
            conf = np.max(lg_mask)

        elif self.search == "greedy":
            # greedy method
            move = legal[np.argmax(lg_mask)]

        if value is not None and value < -0.8:
            move = "resign"

        return move, value, conf


class RandomPlayer():
    def __init__(self):
        pass

    def move(self, game):
        # random moves
        b = game.board
        legal = [x for x in b.legal_moves]
        move = np.random.choice(legal)
        value = None
        conf = None
        return move, value, conf


# test script
if __name__ == "__main__":
    with open("assets/moves.json") as f:
        vocab_size = len(json.load(f))
    config = ModelConfig(
        vocab_size=vocab_size,
        n_positions=180,
        n_ctx=180,
        n_embd=128,
        n_layer=30,
        n_head=8
    )
    player1 = Player(
        config,
        "models/useful/q1/q1_15000.pt",
        "assets/moves.json",
        search="mcts",
        depth=2,
        model_class=BaseHFGPT
    )  # assume to be white
    config = ModelConfig(
        vocab_size=vocab_size,
        n_positions=180,
        n_ctx=180,
        n_embd=128,
        n_layer=30,
        n_head=8
    )
    player2 = Player(
        config,
        "models/useful/q1/q1_15000.pt",
        "assets/moves.json",
        search="sample",
        depth = 2,
        model_class = BaseHFGPT
    ) # assume to be black

    for round in trange(1):
        verbose_print(f"Starting round: {round}", verbose = True)
        game = GameEngine()
        pgn_writer = chess.pgn.Game()
        pgn_writer_node = pgn_writer
        pgn_writer.headers["Event"] = "Test with p=0.95"
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
                verbose_print(mv, "|", m, v, c, verbose = True)

                if m != "resign":
                    done, res = game.step(m)
                    pgn_writer_node = pgn_writer_node.add_variation(m)
                    if res != "game" or mv == 25:
                        verbose_print("Game over", verbose = True)
                        break
                else:
                    verbose_print("Player has resigned", verbose = True)
        except KeyboardInterrupt:
            break
        verbose_print("Saving...", verbose = True)
        verbose_print(pgn_writer, file=open("auto_tournaments_search_d1.pgn", "a"), end="\n\n", verbose = True)
