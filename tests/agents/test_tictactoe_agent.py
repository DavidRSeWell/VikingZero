import numpy as np

from vikingzero.agents.tictactoe_agents import RandomTicTacToeAgent,TicTacToeMinMax,HumanAgent,TicTacToeMCTS
from vikingzero.designers.connect4_designer import Connect4Designer
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

    env = TicTacToe()

    agent1_config = {
        'agent': TicTacToeMCTS,
        'player':1,
        'n_sim': 100,
        'c': 1
        #'type':"alphabeta"
    }

    agent2_config = {
        'agent': RandomTicTacToeAgent,
        #'player':2,
        #'type': "alphabeta"
    }

    designer = Connect4Designer(iters=50,env=env
                                ,agent1_config=agent1_config
                                ,agent2_config=agent2_config
                                )

    designer.run(render=True)


if __name__ == "__main__":

    '''
    for seed in range(1000):
        try:
            test_random_agent(render=False,seed=seed)
        except:
            print(f"Excetiption with seed {seed}")
    '''

    #test_random_agent(seed=2)

    test_designer()
