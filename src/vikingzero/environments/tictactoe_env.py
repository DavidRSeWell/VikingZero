import gym
import numpy as np


class TicTacToe(gym.Env):
    """
    This class handles all the logic of running a tic tac
    toe game. It expects as input two players which themselves
    are expected to have an "act" method that will return the
    action the player wants to take. The players can either
    be bots or humans
    """
    metadata = {'render.modes': ['human']}

    def __init__(self,display_board=False):

        self._display_board = display_board

        self.action_size = 9
        self.current_player = 1
        self.board = np.zeros(9,)
        self.name = "TicTacToe"

        def run_test():
            board = [[2., 2., 1.],
                    [0., 2., 1.],
                    [0.,1.,0.]]

            #self.board = np.array(board).flatten()

            #self.board[0] = 1
            #self.board[1] = 2
            self.board[0] = 2
            self.board[3] = 2
            #self.board[4] = 2
            self.board[5] = 1
            self.board[8] = 1
            self.current_player = 1

        #run_test()

        self.board_string = """
                | {s1} | {s2} | {s3} |    
                 ------------
                | {s4} | {s5} | {s6} |    
                 ------------
                | {s7} | {s8} | {s9} |  
            """
        self.winner = None

    def __str__(self):

        return self.render()

    def is_legal(self,action):
        """
        Checks to make sure an action is legally possible
        :return:
        """

        legal_moves = np.where(self.board == 0.0)[0]
        if action not in legal_moves:
            print("Action {} is not legal action".format(action))
            print("CURRENT BOARD")
            print(self.render())
            return False

        else:
            return True

    def next_state(self, state, action):

        player = self.check_turn(state)
        actions = TicTacToe.actions(state)
        if len(actions) == 0:
            print("The current state is a terminal state cannot get next state")
            return state

        next_s = state.copy()
        next_s[action] = player
        return next_s

    def reset(self):
        """
        Reset the board
        :return:
        """
        self.board = np.zeros(9,)

        def run_test():
            board = [[2., 2., 1.],
                          [0., 2., 1.],
                          [0., 1., 0.]]

            #self.board = np.array(board).flatten()
            #self.board[0] = 1
            # self.board[1] = 2
            self.board[0] = 2
            self.board[3] = 2
            #self.board[4] = 2
            self.board[5] = 1
            self.board[8] = 1
            self.current_player = 1

        self.current_player = 1

        #run_test()

    def render(self, mode="human"):
        board_dict = {f"s{i}": int(self.board[i - 1]) for i in range(1, 10)}
        board_str = str(self.board_string.format(**board_dict))
        print(board_str)
        return board_str

    def step(self,action):
        """
        Take in an action from a player
        and update the game state
        :param action: integer of action
        :return:
        """

        curr_state = self.board.copy()

        if self.is_legal(action):

            next_state = curr_state.copy()

            next_state[action] = self.current_player

            self.board = next_state

            winner = self.check_winner(self.board)

            if not winner:
                self.current_player = 1 if self.current_player == 2 else 2

            else:
                self.winner = winner

            return curr_state,action,next_state,winner

        else:
            print("The action {} by player {} not legal!!!".format(self.current_player,action))
            return -1

    @staticmethod
    def actions(board):
        return np.where(board == 0.0)[0]

    @staticmethod
    def valid_actions(board):
        if TicTacToe.is_win(board):
            return []
        return np.where(board == 0.0)[0]

    @staticmethod
    def check_winner(board):

        # check if this is a winning move
        winner = TicTacToe.is_win(board)
        return winner
        """
        if winner:
            if winner == 1:
                return (1,-1) , winner
            elif winner == 2:
                return (-1,1) , winner
            elif winner == -1:
                return (0,0), winner
        else:
            # game still going
            return (0,0) , winner
        """

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
    def is_draw(board):

        zero_count = len(np.where(board == 0)[0])
        if zero_count == 0 and TicTacToe.is_win(board) == -1:
            return True
        else:
            return False

    @staticmethod
    def is_win(board):
        """
        Check if the current board is a
        win for a given player or if the
        game is a draw.

        :return:
        """

        board = board.copy()
        board[board == 2] = -1
        board_mat = np.reshape(board, (3, 3))

        board_zeros = np.where(board == 0)[0]

        diag_1 = sum([board_mat[i][i] for i in range(3)])  # check diagnoal
        diag_2 = sum([board_mat[i][3 - i - 1] for i in range(3)])  # check other diagnol
        row_sum = board_mat.sum(axis=1)
        col_sum = board_mat.sum(axis=0)
        if 3 in row_sum or (3 in col_sum):  # row sum
            return 1 # player one won
        elif -3 in row_sum or (-3 in col_sum):
            return 2 # player two wone
        elif (3 in (diag_1, diag_2)):
            return 1
        elif (-3 in (diag_1, diag_2)):
            return 2
        # check draw
        elif len(board_zeros) == 0:
            return -1
        else:
            return 0



