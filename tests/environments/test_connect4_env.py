import numpy as np

from vikingzero.environments.connect4_env import Connect4

def test_connect_4_board():
    '''
    Verify structure of board
    :return:
    '''
    env = Connect4()

    assert type(env.board) == np.ndarray

    assert env.board.shape == (6,7)

def connect_4_step():
    '''
    Test that the environment is changing as expected to
    a given set of actions
    :return:
    '''

    env = Connect4()

    action1 = 1

    env.step(action1)

    board1 = np.zeros((6,7))

    board1[0,1] = 1

    assert env.board == board1

    action2 = 1

    env.step(action2)

    board2 = board1.copy()

    board2[1,1] = 1

    assert env.board == board2

    action3 = 0

    env.step(action3)

    board3 = board2.copy()

    board3[0,0] = 1

    assert env.board == board3

def test_connect_4_winner():
    '''
    Test that the environment accurately decides if a board is in
    a winning position or not
    CASES:
        0: Nobody has played
        1: Horizontal simple
        2: Horizontal complex
        3: Vertical simple
        4: Vertical complex
        5: Diagonal normal simple
        6: Diagonal normal complex
        7: Diagonal flipped V simple
        8: Diagonal flipped V complex
        9: Diagonal flipped H simple
        10: Diagonal flipped H complex
    :return:
    '''

    env = Connect4()

    # CASE 0: Nobody has played
    board = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],

    ])

    winner = env.check_winner(board)

    assert winner == 0

    # CASE 1: Horizontal simple
    board1 = np.array([
        [1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ])

    winner = env.check_winner(board1)

    assert winner == 1

    # CASE 2: Horizontal complex
    board2 = np.array([
        [1, 0, 1, 1, 0, 0, 0],
        [0, 2, 2, 0, 0, 0, 0],
        [0, 1, 2, 2, 2, 2, 1],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],

    ])

    winner = env.check_winner(board2)

    assert winner == 2

    # CASE 3: Vertical Simple
    board3 = np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],

    ])

    winner = env.check_winner(board3)

    assert winner == 1

    # CASE 4: Vertical complex
    board4 = np.array([
        [1, 1, 2, 0, 0, 0, 0],
        [0, 0, 0, 2, 0, 0, 0],
        [0, 0, 0, 2, 0, 0, 0],
        [0, 0, 0, 2, 0, 0, 0],
        [0, 0, 1, 2, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0],

    ])

    winner = env.check_winner(board4)

    assert winner == 2

    # CASE 5: Diagnoal Normal Simple
    board5 = np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],

    ])

    winner = env.check_winner(board5)

    assert winner == 1

    # CASE 1: Horizontal Normal Complex
    board6 = np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [2, 2, 0, 0, 0, 0, 0],
        [0, 1, 2, 0, 2, 0, 0],
        [0, 0, 0, 2, 0, 0, 0],
        [0, 0, 0, 0, 2, 0, 0],
        [0, 0, 1, 0, 0, 1, 0],

    ])

    winner = env.check_winner(board6)

    assert winner == 2

    # CASE 1: Horizontal win
    board = np.array([
        [1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],

    ])

    winner = env.check_winner(board)

    assert winner == 1

    # CASE 1: Horizontal win
    board = np.array([
        [1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],

    ])

    winner = env.check_winner(board)

    assert winner == 1

    # CASE 1: Horizontal win
    board = np.array([
        [1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],

    ])

    winner = env.check_winner(board)

    assert winner == 1

    # CASE 1: Horizontal win
    board = np.array([
        [1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],

    ])

    winner = env.check_winner(board)

    assert winner == 1

    # CASE: Draw
    board = np.array([
        [1, 1, 2, 1, 2, 1, 2],
        [2, 1, 2, 1, 1, 2, 1],
        [1, 1, 2, 2, 2, 1, 1],
        [2, 2, 1, 1, 1, 2, 2],
        [1, 2, 1, 2, 2, 1, 2],
        [2, 1, 1, 2, 1, 2, 1],

    ])

    winner = env.check_winner(board)

    assert winner == -1

def test_connect_4_valid_actions():

    env = Connect4()

    # CASE 0: Nobody has played
    board = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],

    ])

    valid_actions = env.valid_actions(board)

    assert len(valid_actions) == 7

    # CASE 1: First Column gone
    board = np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],

    ])

    valid_actions = env.valid_actions(board)

    assert len(valid_actions) == 6

    # CASE 3: None
    board = np.array([
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],

    ])

    valid_actions = env.valid_actions(board)

    assert len(valid_actions) == 0

    # CASE 4:

    board = np.array([[1., 0., 0., 0., 0., 0., 0.],
                    [2., 0., 0., 0., 0., 0., 0.],
                    [1., 0., 0., 0., 0., 0., 0.],
                    [2.0, 2.0,1.0,0.,0.,0.,0.],
                    [2., 1., 1., 0., 1., 0., 0.],
                    [1.,2.,1.,2.,2.,0.,0.]])

    valid_actions = env.valid_actions(board)

    print(valid_actions)
    assert len(valid_actions) == 6

    print("PASSED Valid actions")


if __name__ == "__main__":
    test_connect_4_valid_actions()

