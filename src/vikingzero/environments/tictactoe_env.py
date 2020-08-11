import gym
import numpy as np


class TicTacToe:
    """
    This class handles all the logic of running a tic tac
    toe game. It expects as input two players which themselves
    are expected to have an "act" method that will return the
    action the player wants to take. The players can either
    be bots or humans
    """

    def __init__(self,display_board=False):

        self._display_board = display_board
        self.current_player = 1
        self.board = np.zeros(9,)
        self.board_string = """
                | {s1} | {s2} | {s3} |    
                 ------------
                | {s4} | {s5} | {s6} |    
                 ------------
                | {s7} | {s8} | {s9} |  
            """

        self.pieces = {" ": 0, "X": 1, "O": -1} # used in converting the board dictionary to an array

    def __str__(self):

        return self.render()

    def convert_board_to_array(self):
        """
        Converts the board dictionary to
        a numpy array
        :param board_state: dictionary
        :return:
        """

        board = np.zeros((9,))
        i = 0
        for k, v in self.board_state.items():
            board[i] = self.pieces[v]
            i += 1

        return board

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

    def is_draw(self):

        zero_count = len(np.where(self.board == 0)[0])

        if zero_count == 0:
            return True
        else:
            return False

    def is_win(self):
        """
        Check if the current board is a
        win for a given player or if the
        game is a draw.

        :return:
        """

        board_mat = np.reshape(self.board,(3,3))

        diag_1 = np.abs(sum([board_mat[i][i] for i in range(3)]))  # check diagnoal
        diag_2 = np.abs(sum([board_mat[i][3 - i - 1] for i in range(3)]))  # check other diagnol

        if 3 in np.abs(board_mat.sum(axis=1)): # row sum
            return True

        elif 3 in np.abs(board_mat.sum(axis=0)): # col sum
            return True

        elif (3 in (diag_1,diag_2)):
            return True

        else:
            return False

    def reset(self):
        """
        Reset the board
        :return:
        """
        self.board_state = {"s1": " ", "s2": " ", "s3": " "
            , "s4": " ", "s5": " ", "s6": " ",
                            "s7": " ", "s8": " ", "s9": " "}

        self.game_state = {
            "player": self.player1,  # whos turn is it anyways?
            "win_state": 0  # 0 if game is still going 1 if player "X" won and -1 if player "O" won 2 for a draw
        }

        self.player1.current_state = self.board_state.copy()
        self.player2.current_state = self.board_state.copy()

    def render(self):
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
            reward = self.check_winner()
            if not reward:
                self.current_player*=-1
            else:
                self.winner = self.current_player
            return curr_state,action,next_state,reward

        else:
            print("The action {} by player {} not legal!!!".format(self.current_player,action))
            return -1

    def check_winner(self):

        # check if this is a winning move
        if self.is_win():
            return 1
        elif self.is_draw():
            r = -1
        else:
            r = 0

        return r
