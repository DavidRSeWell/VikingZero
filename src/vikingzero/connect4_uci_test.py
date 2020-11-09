"""
This module was intented to play around with the idea of using
Connect4 Expert data as a way to test validity of algorithms on
the connect4 Environment. In early versions of AlphaGo Expert data was
used as a way to "seed" the NN. Expert Data can also be a good way
to do debugging and testing
"""


import numpy as np
import torch.nn as tnn
import torch.nn.functional as F


def display_board(board):
    columns = 7
    rows = 6

    board = board.astype(np.int)

    def print_row(values, delim="|"):
        return f"{delim} " + f" {delim} ".join(str(v) for v in values) + f" {delim}\n"

    row_bar = "+" + "+".join(["---"] * columns) + "+\n"
    out = row_bar
    for r in range(rows):
        out = out + \
            print_row(board[r * columns: r * columns + columns]) + row_bar

    return out


def convert_to_kaggle(x):
    '''
    Function to take the UCI format for connect 4
    and convert it to kaggle style.
    :param x:
    :return:
    '''

    new_x = np.zeros((6,7))
    index = 0
    for col_i in range(7):
        col = []
        for row_i in range(5, -1, -1):
            new_x[row_i][col_i] = int(x[index])
            index += 1

    new_list = []
    for row_i in range(6):
        for col_i in range(7):
            new_list.append(new_x[row_i][col_i])

    return np.array(new_list)


def process_data(data_path):
    '''
    Read in .data file and return structured
    data in numpy array
    :param data_path:
    :return:
    '''
    data_file = open(data_path,'r')

    data = data_file.readlines()

    X_data, Y_data = [],[]
    for i , d in enumerate(data):

        a = d.replace('b', '0').replace('x', '1').replace('o', '2').replace('\n','').split(',')
        X = a[:-1]
        X = convert_to_kaggle(X)
        Y = -1.0
        if 'win' in a[-1]:
            Y = 1.0
        elif 'draw' in a[-1]:
            Y = 0.0
        X_data.append(X)
        Y_data.append(Y)

    return np.array(X_data).astype(float),np.array(Y_data).reshape((len(Y_data),1))


# this is one way to define a network
class Net(tnn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.h1 = tnn.Linear(n_feature, n_hidden)   # hidden layer
        self.h2 = tnn.Linear(n_hidden,n_hidden*2)
        self.h3 = tnn.Linear(n_hidden*2,n_hidden)
        self.predict = tnn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.h1(x))      # activation function for hidden layer
        x = F.relu(self.h2(x))      # activation function for hidden layer
        x = F.relu(self.h3(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x

