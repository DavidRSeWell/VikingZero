import numpy as np
import pickle

from vikingzero.agents.connect4_agent import RandomConnect4Agent,Connect4MinMax,Connect4MCTS
from vikingzero.designers.connect4_designer import Connect4Designer
from vikingzero.environments.connect4_env import Connect4


def test_random_agent():

    env = Connect4()

    agent1 = RandomConnect4Agent(env)
    agent2 = RandomConnect4Agent(env)

    curr_player = agent1

    while True:

        action = curr_player.act(env.board)

        curr_state,action,next_state,winner = env.step(action)

        print(f"Action = {action}")
        print(f"-----------NEW BOARD-----------")
        env.render()

        if winner != 0:
            print(f"Game over! Winner is {winner}")
            break

        curr_player = agent2

    assert winner in [-1,0,1,2]


def test_designer():

    env = Connect4()

    agent1_config = {
        "agent": "Connect4MinMax",
        "player": 1,
        #"c": np.sqrt(2),
        #"n_sim": 200
        "type": "alphabeta_depth",
        "depth":2,
        "n_sims": 5
    }
    agent2_config = {
        "agent": "Connect4MinMax",
        "player":2,
        "type":"alphabeta_depth",
        "depth":2,
        "n_sims":5
    }



    designer = Connect4Designer(iters=50,env=env
                                ,agent1_config=agent1_config
                                ,agent2_config=agent2_config
                                )

    designer.run(render=True)

    agent1_path = ""
    agent2_path = ""
    # save
    f = open(f"{agent1_path}","wb")
    f2 = open(f"{agent2_path}","wb")
    pickle.dump(designer.agent1,f)
    pickle.dump(designer.agent2,f2)


