"""server file for playing with NeuraPlayer
5.12.2020 - @yashbonde"""

import json
from flask import request, make_response, render_template, Flask, jsonify

from game import Player, RandomPlayer, GameEngine, Move
from model import BaseHFGPT, ModelConfig

# make the app and run the server
app = Flask(__name__)

game = GameEngine()
# player = RandomPlayer()

# define the NeuraPlayer
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
player = Player(
    config,
    "models/useful/q1/q1_15000.pt",
    "assets/moves.json",
    search="sample",
    depth=2,
    model_class=BaseHFGPT
)

@app.route('/move')
def make_move():
    # let the player 
    move_dict = request.args
    move = f"{move_dict['from']}{move_dict['to']}"

    # handling for promotion
    if move_dict["piece"] == "p" and move[-1] in ["1", "8"]:
        move = move + "q"
    move = Move(move)
    done, res = game.step(move)
    print("move", move, game.board.fen())
    
    if done:
        response = make_response(jsonify(content = res))
        return response

    # player makes the move and board gets updated
    move, _, _ = player.move(game)
    game.step(move)
    move = str(move)

    # response
    response = make_response(jsonify(
        board_state=game.board.fen(),
        from_square = move[:2],
        to_square = move[2:]
    ))
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response


@app.route("/")
def new_game():
    global game
    game.reset()
    return render_template("index.html")

# --mode=sf/neura
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
