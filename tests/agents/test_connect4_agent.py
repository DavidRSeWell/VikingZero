
from src.agents.connect4_agent import RandomConnect4Agent
from src.designer.connect4_designer import Connect4Designer
from src.environments.connect4_env import Connect4


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
        'agent': RandomConnect4Agent
    }

    agent2_config = {
        'agent': RandomConnect4Agent
    }

    designer = Connect4Designer(iters=1000,env=env
                                ,agent1_config=agent1_config
                                ,agent2_config=agent2_config
                                )

    designer.run()



if __name__ == "__main__":

    test_designer()






