"""
Module to assist in creating an instantiated class based on
passed parameters from an end user
"""
import numpy as np

from .environments.tictactoe_env import TicTacToe
from .agents.tictactoe_agents import TicTacToeMinMax


def load_agent(name):

    module_name = "vikingzero.agents"
    if "tictactoe" in name.lower():
        module_name += ".tictactoe_agents"

    elif "connect4" in name.lower():
        module_name += ".connect4_agent"

    elif "zero" in name.lower():
        module_name += ".alphago"

    print(f"Loading agent {name}")
    mod = __import__(module_name, fromlist=[name])
    klass = getattr(mod, name)
    return klass


def load_env(name):
    module_name = "vikingzero.environments"
    if "tictactoe" in name.lower():
        module_name += ".tictactoe_env"

    elif "connect4" in name.lower():
        module_name += ".connect4_env"

    mod = __import__(module_name, fromlist=[name])
    klass = getattr(mod, name)
    return klass


def generate_tictactoe_state_dict():

    from .agents.tictactoe_agents import TicTacMCTSNode

    root_board = np.zeros((3,3))

    root_node = TicTacMCTSNode(TicTacToe(), root_board.flatten(), 1, 0, root=True)

    states = {}

    def process_node(node):

        # print("process node")
        # print(node.board)

        if hash(node) not in states:
            states[hash(node)] = node.board

        if node.is_terminal():
            # print("Terminal")
            return

        for child in node.get_children():
            # print("Childe")
            # print(child.board)
            process_node(child)

    process_node(root_node)

    return states


def create_minimax_lookup(save_path,state_dict_path=""):

    env = TicTacToe()

    minimax_agent = TicTacToeMinMax(env,type="alphabeta")

    if len(state_dict_path) > 0:
        try:
            states = np.load(state_dict_path,allow_pickle=True).item()
        except:
            print("State dict path does not appear to be correct. Creating new state dict")
            pass

    else:
        states = generate_tictactoe_state_dict()

    minimax_actions = {}

    for k,v in states.items():

        env.current_player = env.check_turn(v)

        valid_actions = env.valid_actions(v)

        if len(valid_actions) == 0:
            continue

        a = minimax_agent.act(v)

        minimax_actions[k] = list(a)

    print("Done creating minimax lookup")
    print("Num states .....")
    print(len(minimax_actions))
    np.save(save_path,minimax_actions)










