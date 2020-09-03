import torch.nn as nn
import torch.nn.functional as F

class UCINet(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(UCINet, self).__init__()
        self.h1 = nn.Linear(n_feature, n_hidden)   # hidden layer
        self.h2 = nn.Linear(n_hidden,n_hidden*2)
        self.h3 = nn.Linear(n_hidden*2,n_hidden)
        self.predict = nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.h1(x))      # activation function for hidden layer
        x = F.relu(self.h2(x))      # activation function for hidden layer
        x = F.relu(self.h3(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x

