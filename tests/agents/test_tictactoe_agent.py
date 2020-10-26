import numpy as np

from vikingzero.agents.tictactoe_agents import RandomTicTacToeAgent,TicTacToeMinMax,HumanAgent\
    ,TicTacToeMCTS, TicTacToeMCTS_NN
from vikingzero.designers.connect4_designer import Connect4Designer,Designer
from vikingzero.environments.tictactoe_env import TicTacToe


def test_random_agent(render=True,seed=3):

    np.random.seed(seed)
    env = TicTacToe()

    agent1 = RandomTicTacToeAgent(env)
    agent2 = RandomTicTacToeAgent(env)

    curr_player = agent1

    while True:

        action = curr_player.act(env.board)

        curr_state, action, next_state, winner = env.step(action)

        if render:
            print(f"Action = {action}")
            print(f"-----------NEW BOARD-----------")
            env.render()

        if winner != 0:
            print(f"Game over! Winner is {env.current_player}")
            break

        curr_player = agent2


def test_designer():


    agent_config = {

        "agent1": {

            "agent": "TicTacToeMinMax",
            #"node": TicTacMCTSNode,
            #"n_sim": 50,
            #"batch_size": 15,
            #"c": 1.41,
            "player": 1,
            "type": "minimax"
        },

        "agent2": {

            "agent": "TicTacToeMinMax",
            # "node": TicTacMiniNode,
            # "n_sim": 50,
            # "batch_size": 5,
            # "c": 1.41,
            "player": 2,
            "type": "minimax"
        },

    }

    exp_config = {
        "episodes": 10,
        "record_all": False,
        "record_every": 1,
        "eval_iters": 1,
        "render": True,
        "train_iters": 5
    }


    designer = Designer(TicTacToe(),agent_config,exp_config,_run=False)

    designer.play_game(True,designer.agent1,designer.agent2)


