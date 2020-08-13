import numpy as np

from vikingzero.environments.tictactoe_env import TicTacToe

ACTIONS = [a for a in range(9)]

BOARD_0 = np.zeros(9,)

BOARD_1 = BOARD_0.copy()

DRAWBOARD = BOARD_0.copy()
### HELPER FUNC TO BUILD TEST BOARDS


def create_test_board_1(board):
    board[0] = 1
    board[1] = 1
    board[2] = 2
    board[3] = 2
    board[7] = 1
    board[8] = 2

    return board

def create_draw_board(board):

    board = board.copy()
    board[0] = 1
    board[1] = 2
    board[2] = 2
    board[3] = 2
    board[4] = 1
    board[5] = 1
    board[6] = 1
    board[7] = 2
    board[8] = 2
    return board

def render(board):
    board = board.reshape((3,3))
    print(board)

create_test_board_1(BOARD_1)


def test_islegal():

    env = TicTacToe()
    env.board = BOARD_0

    for a in ACTIONS:
        assert env.is_legal(a)

def test_step():

    env = TicTacToe()
    board1 = create_test_board_1(BOARD_1)
    env.board = board1

    assert env.step(1) == -1

    s0,a , s1 , r = env.step(4)

    board2 = board1.copy()
    board2[4] = 1

    assert np.equal(s0,board1).all()
    assert np.equal(s1,board2).all()
    assert a == 4
    assert r == 1

def test_draw():

    #env = TicTacToe()
    draw_board = create_draw_board(BOARD_0)

    assert TicTacToe.is_draw(draw_board)



    print("Passed test_draw")

def test_win():

    win_board = BOARD_1.copy()
    win_board[4] = 1
    assert TicTacToe.is_win(win_board) == 1

    diag_board = BOARD_0.copy()

    diag_board[0] = 1
    diag_board[4] = 1
    diag_board[8] = 1

    assert TicTacToe.is_win(diag_board) == 1

    diag_board = BOARD_0.copy()

    diag_board[0] = 2
    diag_board[4] = 2
    diag_board[8] = 2

    assert TicTacToe.is_win(diag_board) == 2

    row_board = BOARD_0.copy()
    row_board[0] = 1
    row_board[1] = 1
    row_board[2] = 1

    assert TicTacToe.is_win(row_board) == 1

    row_board = BOARD_0.copy()
    row_board[0] = 2
    row_board[1] = 2
    row_board[2] = 2

    assert TicTacToe.is_win(row_board) == 2

    col_board = BOARD_0.copy()
    col_board[0] = 2
    col_board[3] = 2
    col_board[6] = 2

    assert TicTacToe.is_win(col_board) == 2

    draw_board = create_draw_board(BOARD_0)

    assert TicTacToe.is_win(draw_board) == -1

    assert TicTacToe.is_win(BOARD_0) == 0

    print("Passed test_win")



if __name__ == "__main__":


    #test_win()
    #draw_board = create_draw_board(BOARD_0)

    #render(draw_board)

    #test_step()

    test_draw()

    test_win()




