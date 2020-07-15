
from src.agents.connect4 import RandomConnect4Agent
from src.environments.connect4 import Connect4

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



if __name__ == "__main__":

    test_random_agent()






