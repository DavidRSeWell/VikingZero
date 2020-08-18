import numpy as np

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
        "agent": Connect4MinMax,
        "player":1,
        "type":"alphabeta_depth"
    }

    agent2_config = {
        "agent": Connect4MCTS,
        "player": 2,
        "c":np.sqrt(2),
        "n_sim":200

        #"type": "alphabeta_depth"
    }

    designer = Connect4Designer(iters=100,env=env
                                ,agent1_config=agent1_config
                                ,agent2_config=agent2_config
                                )

    designer.run(render=True)



if __name__ == "__main__":

    test_designer()






