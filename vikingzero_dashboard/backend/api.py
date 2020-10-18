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

envs = {
    "tictactoe": TicTacToe(),
    "connect4": Connect4()

}

minimax_tictactoe_agent = TicTacToeMinMax(envs["tictactoe"],player=2,type="alphabeta")
minimax_connect4_agent = Connect4MinMax(envs["connect4"],player=2,type="alphabeta",depth=2,n_sims=5)

test_path = "/Users/befeltingu/Documents/GitHub/VikingZero/tests/"
tictactoe_config_path = test_path + "tictactoe_alphago.yaml"
connect4_config_path = test_path + "test_alphago.yaml"

def load_agent2(env,config_path,agent_model_name):

    config_file = load_config_file(config_path)

    agent_config = config_file["agent_config"]["agent1"]

    agent_name = agent_config["agent"]

    del agent_config["agent"]

    agent =  AlphaZero(env,**agent_config)

    agent_model_path = "/Users/befeltingu/Documents/GitHub/VikingZero/tests/" + agent_model_name

    agent._nn.load_state_dict(torch.load(agent_model_path))

    agent._act_max = False

    return agent

tictactoe_agent = load_agent2(envs["tictactoe"],tictactoe_config_path,agent_model_name="current_best_TicTacToe_SAN-279")
connect4_agent = load_agent2(envs["connect4"],connect4_config_path,agent_model_name="current_best_Connect4_SAN-216")


def next_state(env_name,state,action,player):

    env = envs[env_name]

    actions = env.actions(state)
    if len(actions) == 0:
        print("The current state is a terminal state cannot get next state")
        return state

    next_s = state.copy()
    if env_name == "connect4":
        next_s = next_s.reshape((6,7))
        row = env.process_move(next_s,action)
        next_s[row,action] = player
    else:
        next_s[action] = player

    return next_s.flatten()

agents = {
    "tictactoe": {
        "minimax": minimax_tictactoe_agent,
        "alphago": tictactoe_agent
    },
    "connect4": {
        "minimax": minimax_connect4_agent,
        "alphago": connect4_agent
    }
}

@app.route("/alpha_opinion", methods=["Post","Get"])
def alpha_opinion():

    data = request.get_json()
    player = int(data["player"])
    board = data["board"]
    board = json_to_board(board)
    env_name = data["env_name"].lower()
    env = envs[env_name]
    board = board.reshape((env.board.shape))
    agent = agents[env_name]["alphago"]
    action_size = agent._action_size
    board_t = agent.transform_board(board)

    board_var = Variable(torch.from_numpy(board_t).type(dtype=torch.float))

    p, v = agent._nn.predict(board_var)

    actions = env.valid_actions(board)

    p[[a for a in range(agent._action_size) if a not in actions]] = 0

    p = p / p.sum()

    curr_board = board.copy()
    value_board = [0 for _ in range(action_size)]

    for a in actions:
        next_board = next_state(env_name,curr_board, a, int(player))
        next_board = agent.transform_board(next_board)
        board_var_a = Variable(torch.from_numpy(next_board).type(dtype=torch.float))
        p_a, v_a = agent._nn.predict(board_var_a)
        value_board[a] = np.round(float(-1.0*v_a),2)

    # Get prob distribution for move
    agent.reset()

    action , p_a = agent.act(board)

    return {"p":p.tolist(), "v": value_board, "mcts_p":p_a.tolist()}

@app.route("/is_win",methods=["Post","Get"])
def is_win():
    data = request.get_json()
    board = data["board"]
    env_name = data["env_name"].lower()
    env = envs[env_name]
    board = json_to_board(board)
    board = board.reshape(env.board.shape)
    win = env.is_win(board)
    return {"win":win} ,201

@app.route("/make_move",methods=["Post","Get"])
def make_move():

    move_data = request.get_json()

    action = move_data["action"]

    board = json_to_board(move_data["board"])

    player = move_data["player"]

    agent_type = move_data["agent"]

    env_name = move_data["env_name"].lower()

    env = envs[env_name]

    agent = agents[env_name][agent_type]

    env.board = board

    env.current_player = player

    curr_state, action, next_s, winner = env.step(action)
    if winner:
        next_board = board_to_json(next_s)
        return {"board": next_board, "winner": winner}, 201

    if agent_type == "alphago":
        agent.reset()
        agent._act_max = True

    a = agent.act(next_s)
    if type(a) == tuple:
        a,p_a = a

    curr_state, action, next_s, winner = env.step(a)

    next_board = board_to_json(next_s)

    return {"board": next_board, "winner":winner} , 201

@app.route("/new_game",methods=["Post"])
def new_game():

    move_data = request.get_json()

    env_name = move_data["env_name"].lower()

    envs[env_name].reset()

    return {}, 201

@app.route("/get_next_state", methods=["Post","Get"])
def get_next_state():

    move_data = request.get_json()

    action = move_data["action"]

    board = json_to_board(move_data["board"])

    env_name = move_data["env_name"].lower()

    if env_name == "connect4":
        action = board_index_to_col(action)

    env = envs[env_name]

    player = env.check_turn(board)

    env.board = board

    next_board = next_state(env_name,board,action,player)

    return {"board": board_to_json(next_board) }, 201


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


def board_index_to_col(action):
    a = np.zeros((42,))
    a[action] = 1
    b = a.reshape((6,7))
    action = int(np.where(b == 1)[1])
    return action


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

def create_json_tree(agent,node):
    """
    From the current node create a json tree that we can
    pass back to front end
    :param node:
    :return:
    """

    children = agent.children[node]

    tree = {
        "name": "0",
        "children": [

        ]
    }

    keys = {
        node: 0
    }
    node_num = 0

    def update_tree(node,parent):
        curr_children = node.get_children()
        if node not in keys:
            node_num += 1
            keys[node] = node_num


        for child in curr_children:
            if child in keys:





    for key,value in agent.children.items():
        pass



