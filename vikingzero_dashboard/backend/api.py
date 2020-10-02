"""
Module for connecting the vikingzero dashboard to the backend
"""
import numpy as np
import torch

from flask import Flask, jsonify, request
from sacred.config.config_dict import ConfigDict
from sacred.config.config_files import load_config_file
from torch.autograd import Variable

from vikingzero.agents.tictactoe_agents import TicTacToeMinMax
from vikingzero.agents.connect4_agent import Connect4MinMax
from vikingzero.agents.alphago import AlphaZero
from vikingzero.environments.tictactoe_env import TicTacToe
from vikingzero.environments.connect4_env import Connect4
from vikingzero.utils import load_agent

app = Flask(__name__)

env = TicTacToe()
#env = Connect4()

minimax_agent = TicTacToeMinMax(env,player=2,type="alphabeta")
#minimax_agent = Connect4MinMax(env,player=2,type="alphabeta",depth=2,n_sims=5)

config_path = "/Users/befeltingu/Documents/GitHub/VikingZero/tests/tictactoe_alphago.yaml"
#config_path = "/Users/befeltingu/Documents/GitHub/VikingZero/tests/test_alphago.yaml"

config_file = load_config_file(config_path)

agent_config = config_file["agent_config"]["agent1"]

agent_name = agent_config["agent"]

del agent_config["agent"]

alphago_agent = AlphaZero(env,**agent_config)

agent_model_path = "/Users/befeltingu/Documents/GitHub/VikingZero/tests/current_best_TicTacToe_SAN-135"
#agent_model_path = "/Users/befeltingu/Documents/GitHub/VikingZero/tests/current_best_Connect4_170"

alphago_agent._nn.load_state_dict(torch.load(agent_model_path))

alphago_agent._act_max = False

agents = {
    "minimax":minimax_agent,
    "alphago":alphago_agent
}

def next_state(state,action,player):
    actions = TicTacToe.actions(state)
    if len(actions) == 0:
        print("The current state is a terminal state cannot get next state")
        return state

    next_state = state.copy()
    next_state[action] = player
    return next_state

@app.route("/alpha_opinion", methods=["Post","Get"])
def alpha_opinion():
    data = request.get_json()
    player = int(data["player"])
    board = data["board"]
    board = json_to_board(board)
    print("BOARD")
    print(data)
    print(board)
    board_var = Variable(torch.from_numpy(board.reshape((1,9))).type(dtype=torch.float))

    p, v = agents["alphago"].predict(board_var)

    print("Original proediction")
    print(p.reshape((3,3)))
    print(np.where(board == 0.0))

    actions = env.valid_actions(board)

    print("Valid actions")
    print(actions)
    p[[a for a in range(agents["alphago"]._action_size) if a not in actions]] = 0
    print(p)
    p = p / p.sum()
    print(p)

    curr_board = board.copy()
    if env.name == "TicTacToe":
        value_board = [0 for _ in range(9)]
    elif env.name == "Connect4":
        value_board = [0 for _ in range(42)]

    for a in actions:
        next_board = next_state(curr_board, a, int(player))
        board_var_a = Variable(torch.from_numpy(next_board.reshape((1,9))).type(dtype=torch.float))
        p_a, v_a = agents["alphago"].predict(board_var_a)
        value_board[a] = np.round(float(-1.0*v_a),2)

    # Get prob distribution for move
    agents["alphago"].reset()
    action , p_a = agents["alphago"].act(board)
    print("returning request")
    print(value_board)
    return {"p":p.tolist(), "v": value_board, "mcts_p":p_a.tolist()}


@app.route("/make_move",methods=["Post","Get"])
def make_move():

    move_data = request.get_json()

    print("Move data")
    print(move_data)

    action = move_data["action"]

    board = json_to_board(move_data["board"])

    player = move_data["player"]

    agent_type = move_data["agent"]

    agent = agents[agent_type]

    env.board = board

    env.current_player = player

    curr_state, action, next_state, winner = env.step(action)
    if winner:
        next_board = board_to_json(next_state)
        return {"board": next_board, "winner": winner}, 201

    if agent_type == "alphago":
        agent.reset()
        agent._act_max = True

    a = agent.act(next_state)
    if type(a) == tuple:
        a,p_a = a

    curr_state, action, next_state, winner = env.step(a)

    next_board = board_to_json(next_state)

    return {"board": next_board, "winner":winner} , 201


@app.route("/new_game",methods=["Post"])
def new_game():

    move_data = request.get_json()


    env.reset()

    return {}, 201


def board_to_json(board):

    board = board.tolist()

    for idx, item in enumerate(board):
        if item == 1:
            board[idx] = "X"
        elif item == 2:
            board[idx] = "O"
        elif item == 0:
            board[idx] = None

    return board


def json_to_board(board):

    board = list(board)

    for idx, item in enumerate(board):
        if item == "X":
            board[idx] = 1
        elif item == "O":
            board[idx] = 2
        elif item == None:
            board[idx] = 0

    return np.array(board)
