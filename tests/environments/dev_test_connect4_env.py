import numpy as np

from src.environments.connect4_env import Connect4

def dev_connect_4_render():

    print('Running connect 4 render method')

    env = Connect4()

    print(env.board)

    print('--------------------------------')


def dev_connect4_check_winner():

    env = Connect4()

    board = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],

    ])

    env.check_winner(board)




if __name__ == '__main__':

    dev_connect_4_render()

    dev_connect4_check_winner()

