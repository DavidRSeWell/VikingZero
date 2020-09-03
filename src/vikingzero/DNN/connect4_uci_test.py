import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as tnn
import torch.nn.functional as F

from torch.autograd import Variable


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


if __name__ == '__main__':

    test_process_data = 0
    if test_process_data:
        print('Processing data')
        X,Y = process_data('/Users/befeltingu/PSUClasses/AdvancedML/connect4.data')
        print(X.shape,Y.shape)
        print('Done processing')

    test_regression = 1
    if test_regression:

        X,Y = process_data('/Users/befeltingu/Documents/GitHub/VikingZero/src/vikingzero/data/connect4.data')

        ###########################
        # CREATE TEST / TRAIN SET
        ###########################
        TRAIN_SPLIT = 0.95
        train_number = int(TRAIN_SPLIT*X.shape[0])
        test_number = X.shape[0] - train_number
        X_indexes = [x_i for x_i in range(X.shape[0])]
        train_indexes = np.random.choice(X_indexes,size=int(train_number),replace=False)
        test_indexes = [index_i for index_i in X_indexes if index_i not in train_indexes]

        def get_test_loss(X_test,Y_test):
            inputs = Variable(torch.from_numpy(X_test).type(dtype=torch.float))

            labels = Variable(torch.from_numpy(Y_test).type(dtype=torch.float))

            with torch.no_grad():
                outputs = model(inputs)

                # get loss for the predicted output
                loss = criterion(outputs, labels)

            return loss

        X_train, Y_train = X[train_indexes],Y[train_indexes]
        X_test, Y_test = X[test_indexes],Y[test_indexes]

        print('Number training {}'.format(X_train.shape[0]))
        print('Number testing {}'.format(X_test.shape[0]))

        inputDim = X.shape[1]  # takes variable 'x'
        outputDim = 1  # takes variable 'y'
        learningRate = 0.002
        hidden_layer_size = 150
        epochs = 30000

        model = Net(inputDim,hidden_layer_size,outputDim)

        criterion = torch.nn.MSELoss()
        #criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learningRate,momentum=0.9)
        train_losses = []
        test_losses = []

        for epoch in range(epochs):
            # Converting inputs and labels to Variable
            inputs = Variable(torch.from_numpy(X_train).type(dtype=torch.float))

            labels = Variable(torch.from_numpy(Y_train).type(dtype=torch.float))

            # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
            optimizer.zero_grad()

            # get output from the model, given the inputs
            outputs = model(inputs)

            # get loss for the predicted output
            loss = criterion(outputs, labels)

            # get gradients w.r.t to parameters
            loss.backward()

            # update parameters
            optimizer.step()

            if (epoch % 500) == 0:

                train_losses.append(loss.item())

                test_loss = get_test_loss(X_test,Y_test)

                test_losses.append(test_loss)

                print('epoch {}, loss {}'.format(epoch, loss.item()))
                print(f'test loss = {test_loss}')

        plt.plot(train_losses,label='train')
        plt.plot(test_losses,label='test')
        plt.legend()
        plt.show()

        inputs = Variable(torch.from_numpy(X_test).type(dtype=torch.float))

        labels = Variable(torch.from_numpy(Y_test).type(dtype=torch.float))



        torch.save(model.state_dict(), '/Users/befeltingu/Documents/GitHub/VikingZero/src/vikingzero/data/model_weights')

        '''
        #####################
        # Run test
        #####################
        inputs = Variable(torch.from_numpy(X_train).type(dtype=torch.float))
        labels = Variable(torch.from_numpy(Y_train).type(dtype=torch.float))
        with torch.no_grad():
            outputs = model(inputs)
            # get loss for the predicted output
            loss = criterion(outputs, labels)
        '''

    run_load_model = 0
    if run_load_model:

        inputDim = 42  # takes variable 'x'
        outputDim = 1  # takes variable 'y'
        learningRate = 0.01
        hidden_layer_size = 100

        model = Net(inputDim,hidden_layer_size,outputDim)

        model.load_state_dict(torch.load('data/model_weights'))

        model.eval()

        X,Y = process_data('/Users/befeltingu/PSUClasses/AdvancedML/connect4.data')

        inputs = Variable(torch.from_numpy(X).type(dtype=torch.float))

        labels = Variable(torch.from_numpy(Y).type(dtype=torch.float))

        #with torch.no_grad():
        #    outputs = model(inputs)

            # get loss for the predicted output
            #loss = criterion(outputs, labels)


        good_count = 0
        for index,x_i in enumerate(X):

            #tensor_x_i = Variable(torch.from_numpy(X[index]).type(dtype=torch.float))
            tensor_x_i = torch.from_numpy(X[index]).type(dtype=torch.float)

            board = np.load('data/board_test.npy')
            tensor_x_i_test = torch.from_numpy(board).type(dtype=torch.float)
            predict_test = model(tensor_x_i_test)

            board = np.zeros(42)
            board[-3] = 1.0
            board[-7] = 2.0

            tensor_x_i_test = torch.from_numpy(board).type(dtype=torch.float)
            predict_test = model(tensor_x_i_test)

            num_actions = len(torch.where(tensor_x_i != 0)[0])

            predict = model(tensor_x_i).detach().numpy()[0]
            actual = Y[index]
            diff = np.abs(predict - actual)
            bad_guess_count =0
            if predict >  1.5 and num_actions < 5:
                print('Prediction = {}'.format(predict))
                print('Actual = {}'.format(Y[index]))
                print(display_board(X[index]))

                #print('pause bros')
                bad_guess_count += 1


        print('Bad guess count = {}'.format(bad_guess_count))
