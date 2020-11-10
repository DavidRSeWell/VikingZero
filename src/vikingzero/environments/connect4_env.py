import numpy as np
import gym


class Connect4(gym.Env):

    metadata = {"render.modes": ["human"]}

    def __init__(self,display_board=False):

        self._display_board = display_board

        self.action_size = 7

        self.board = np.zeros((6,7))

        self.current_player = 1

        self.name = "Connect4"

        self.winner = None

    def close(self):
        self.close()

    def process_move(self,board,action):
        """
        Compute row of action that user took
        :param board:
        :param action:
        :return:
        """
        legal_play = np.where(board[:, action] == 0)[0]

        if len(legal_play) == 0:
            print("No legal move")
            print("BOARD")
            print(board)
            print(action)
            raise Exception(f"Illegal move {action}")

        else:
            return legal_play[-1]

    def next_state(self,board,action):

        curr_state = board.copy()

        board_row = self.process_move(curr_state, action)

        next_state = curr_state.copy()

        next_state[board_row, action] = self.check_turn(board)

        return next_state

    def reset(self):
        self.board = np.zeros((6, 7))
        self.current_player = 1
        self.winner = None

    def render(self, mode="human"):
        print(self.board)

    def step(self,action):
        """
        :param action: Integer of column number to place piece
        :return:
        """

        curr_state = self.board.copy()

        board_row = self.process_move(curr_state,action)

        next_state = curr_state.copy()

        next_state[board_row, action] = self.current_player

        winner = self.check_winner(next_state)

        if winner:
            self.winner = winner

        self.board = next_state

        self.current_player = 1 if self.current_player == 2 else 2

        return curr_state,action,next_state,winner

    @staticmethod
    def check_diagnol(board):

        # check diagonal top left to bottom right
        for i in range(board.shape[0]):
            diag = np.diagonal(board, i, axis1=1, axis2=0)

            diag = diag.reshape((1, len(diag)))

            c = Connect4.check_horizontal(diag)

            if c: return c

        # check diagonal top right to bottom left
        for j in range(board.shape[1]):

            diag = np.diagonal(board, j)

            diag = diag.reshape((1, len(diag)))

            c = Connect4.check_horizontal(diag)

            if c: return c

        return 0

    @staticmethod
    def check_horizontal(board):

        n = board.shape[0]

        m = board.shape[1]

        # check horizontal
        for i in range(n):
            curr_piece = board[i][0]
            count = 1
            for j in range(1, m):

                next_piece = board[i][j]

                if (curr_piece == 0) or (curr_piece != next_piece):
                    count = 1
                    curr_piece = next_piece
                    continue

                if curr_piece == next_piece:
                    count += 1
                    curr_piece = next_piece
                    if count == 4:
                        return curr_piece
                    continue

        return 0

    @staticmethod
    def check_winner(board: np.array) -> int:
        """
        Check if board contains a winner or draw
        returns () if game is not complete
        :param board:
        :return:
        """

        # TODO I threw up a little when I saw all these if statements I wrote

        # check horizontal
        c = Connect4.check_horizontal(board)

        if c:
            return c

        # check vertical
        c = Connect4.check_horizontal(board.T)

        if c:
            return c

        c = Connect4.check_diagnol(board)

        if c:
            return c

        # flip vertically and check
        c = Connect4.check_diagnol(np.flip(board))

        if c:
            return c

        # flip horizontally and check
        c = Connect4.check_diagnol(np.flip(board,axis=1))

        if c:
            return c

        # check for draw
        num_zeros = np.where(board == 0)

        if len(num_zeros[0]) == 0:
            # we have played to this point without a winner and board is full
            return -1

        # no winners
        return 0

    @staticmethod
    def check_turn(board):
        """
        Check whos turn it is to act
        :param board:
        :return:
        """
        count_1 = len(np.where(board.flatten() == 1)[0])
        count_2 = len(np.where(board.flatten() == 2)[0])

        if count_1 > count_2:
            return 2
        else:
            return 1

    @staticmethod
    def is_win(board):
        #TODO Get rid of this by synch methods with tictactoe
        return Connect4.check_winner(board)

    @staticmethod
    def valid_actions(s: np.array) -> np.array:
        """
        :param s: numpy array of the state
        :return: numpy array representing the set of valid actions
        """

        if Connect4.is_win(s):
            return []

        if len(s.shape) == 1:
            s = s.reshape((6,7))
        return np.where(s[0,:] == 0)[0]

    @staticmethod
    def actions(s: np.array) -> np.array:
        return Connect4.valid_actions(s)

