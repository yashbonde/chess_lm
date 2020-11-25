"""main runner file for our chess engine setup
13.11.2019 - @yashbonde"""

import os
import uuid
import random
import logging
from flask import request, make_response
from flask import Flask, jsonify

from game import Player, GameEngine

# make the app and run the server
app = Flask(__name__)

game = GameEngine()
print("INIT GAME:", game)
# neuraPlayer = Player()

@app.route('/move', methods = ["POST"])
def make_move():
    # # get payload data
    # payload = request.get_json()
    print("\n\n", request.args)
    # auth_token = payload["auth_token"]
    # game_id = payload['game_id']
    # from_ = payload['from']
    # to_ = payload['target']
    # player_no = payload['player_no']
    # board_fen = payload['board_fen']
    # san = payload['san'] # this is the notation we would like to save}

    # # update move in DB for current player and game state if required
    # moves.add_move_using_auth(CURR, CONN, game_id, auth_token, board_fen, san)
    # if san[-1] == '#': # this means that the game has ended
    #     games.end_game(CURR, CONN, game_id)
    #     return make_response(jsonify(
    #         board_state = board_fen,
    #         from_square = None,
    #         to_square = None,
    #         content = 'Checkmate, You Won! Proceed to a new game, my child!'
    #     ))

    # # feed the AI new state and get legal moves
    # res = move_orchestrator(board_fen)

    # # update the move for the opponent plus update the game state if needed
    # moves.add_opp_move_using_auth(CURR, CONN, player_no, game_id, auth_token, res['new_state'], res['san'])
    # if res['content'] is not None:
    #     games.end_game(CURR, CONN, game_id)
    #     games.end_game(CURR, CONN, game_id)
    #     return make_response(jsonify(
    #         board_state = board_fen,
    #         from_square = None,
    #         to_square = None,
    #         content = res['content']
    #     ))

    # # return the new move
    # return make_response(jsonify(
    #     board_state = res['new_state'],
    #     from_square = res['from'].lower(),
    #     to_square = res['to'].lower(),
    #     content = res['content']
    # ))

    response = make_response(jsonify(
        board_state="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
    ))
    response.headers["Access-Control-Allow-Origin"] = "*"
    print(response)
    return response


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
