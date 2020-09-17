import copy
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import namedtuple,defaultdict
from dataclasses import dataclass
from operator import itemgetter
from torch.autograd import Variable

from ..search import MCTS
from ..designers.connect4_designer import Designer
from ..agents.tictactoe_agents import TicTacMCTSNode

Memory = namedtuple('Transition',
                        ('state', 'action', 'action_dist', 'value', 'z'))

@dataclass
class Memory:
    state: np.array
    action_dist: np.array
    action_mcts: np.array
    value: np.float
    z: np.float


class Network(nn.Module):

    def __init__(self):
        super().__init__()

        # This represents the shared layer(s) before the different heads
        # Here, I used a single linear layer for simplicity purposes
        # But any network configuration should work
        self.h1 = nn.Linear(9, 18)
        self.h2 = nn.Linear(18, 18)
        self.h3 = nn.Linear(18, 9)
        #self.h4 = nn.Linear(18, 9)

        # Set up the different heads
        # Each head can take any network configuration
        self.policy = nn.Linear(9 , 9)
        self.value = nn.Linear(9, 1)

    def forward(self, x):

        # Run the shared layer(s)
        x = F.relu(self.h1(x))
        x = F.relu(self.h2(x))
        x = F.relu(self.h3(x))
        #x = F.relu(self.h4(x))

        # Run the different heads with the output of the shared layers as input
        policy_out = F.log_softmax(self.policy(x), dim=1)
        value_out = torch.tanh(self.value(x))

        return policy_out, value_out


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, mem):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = mem
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        data = random.sample(self.memory, batch_size)
        return [(d.state,d.action_mcts,d.z) for d in data]

    def save(self,path):

        for d in self.memory:
            print("---------BOARD-------------")
            board = d.state.reshape((3,3))
            print(board)
            print("Result value")
            print(f"Z = {d.z}")
            print("Action Probs")
            print(f"{d.action_mcts}")


    def __len__(self):
        return len(self.memory)


class AlphaZero(MCTS):

    def __init__(self,env, n_sim: int = 50, batch_size: int = 10,max_mem_size: int = 1000,
                 epochs: int = 10, c: int = 1, lr: float = 0.001,
                 player: int = 1):
        super().__init__(c)

        self.player = player

        self._action_size = 9
        self._act_max = False
        self._batch_size = batch_size
        self._current_memory = [] # hold memory for current game
        self._env = env
        self._epochs = epochs
        self._max_mem_size = max_mem_size
        self._memory = self.create_memory()
        self._nn = self.create_model()
        self._n_sim = n_sim
        self._optimizer = torch.optim.Adam(self._nn.parameters(),lr=lr)
        #self._optimizer = torch.optim.SGD(self._nn.parameters(),lr=0.01,momentum=0.9)
        self._softmax = torch.nn.Softmax(dim=1)
        self._tau = 1

        self._v_loss = torch.nn.MSELoss(reduction='sum')

        self._Node = TicTacMCTSNode

    def act(self,board):

        s = self._Node(self._env,board,self._env.current_player,0)
        # First rum simulations to collect
        # Tree statistics
        for _ in range(self._n_sim):

            self.run(s)

        a,p_a = self.get_action(s)

        if len(board.shape) == 1:
            board = board.reshape((1, board.shape[0]))

        #board = Variable(torch.from_numpy(board).type(dtype=torch.float))

        #p, v = self._nn(board)

        memory = Memory(board,None,p_a,None,None)

        self._current_memory.append(memory)

        return a

    def create_model(self):
        return Network()

    def create_memory(self):
        return ReplayMemory(self._max_mem_size)

    def get_action(self,s) -> tuple:
        """
        Sample action based on statistics from tree
        :param s:
        :return:
        """

        #TODO Optim
        temp_power = 1.0 / self._tau

        children = self.children[s]

        actions = [self.get_parent_action(s,c) for c in children]

        c_counts = np.array([self._N[c]**temp_power for c in children])

        c_sum = sum(c_counts)

        p_a = c_counts / c_sum

        p = np.zeros(self._action_size)

        p[actions] = p_a

        child_act = np.random.choice(children, p=p_a)

        if self._act_max:
            # view network
            net_view = np.array([float(self.node_to_board(c)[1]) for c in children])
            counts = np.array([self._N[c] for c in children])
            c = np.zeros(self._action_size)
            v = c.copy()
            c[actions] = counts
            v[actions] = net_view
            print(c.reshape((3,3)))
            print(v.reshape((3,3)))
            child_act = children[np.argmax(p_a)]

        return self.get_parent_action(s,child_act), p

    def get_parent_action(self,parent,child):

        a = parent.board

        b = child.board

        diff = a - b
        diffs = np.where(diff != 0)

        action = diffs[0][0]

        return action

    def node_to_board(self,s):
        board = s.board
        if len(board.shape) == 1:
            board = board.reshape((1, board.shape[0]))
        board = Variable(torch.from_numpy(board).type(dtype=torch.float))

        return self._nn(board)

    def reset_current_memory(self):
        self._current_memory = []

    def reset_memory(self):
        self._memory.memory = []
        self._memory.position = 0

    def reset_tree(self):
        self._Q = defaultdict(int)
        self._N = defaultdict(int)

        self.children = dict()

    def select_child(self,node):

        children = self.children[node]

        board = node.board

        if len(board.shape) == 1:
            board = board.reshape((1, board.shape[0]))

        board = Variable(torch.from_numpy(board).type(dtype=torch.float))

        p , v = self._nn(board)

        p = p.detach().numpy()
        # renormalize
        p = p / p.sum()

        p = p.reshape(self._action_size)

        N_p = self._N[node]

        child_v = []
        for child in children:
            a = self.get_parent_action(node, child)
            p_a = p[a]
            u_c = self._Q[child] / (self._N[child] + 1) + self.c * p_a * (np.sqrt(N_p) / (1 + self._N[child]))
            child_v.append((child, u_c))

        # return max(self.children[node], key=uct)

        return max(child_v, key=itemgetter(1))[0]

    def simulate(self,node):

        if node.is_terminal():
            reward = node.reward()
            return (node.winner, reward)

        board = node.board
        if len(board.shape) == 1:
            board = board.reshape((1,board.shape[0]))
        board = Variable(torch.from_numpy(board).type(dtype=torch.float))

        p,v = self._nn(board)

        return (node.player,float(v))

    def store_memory(self,z):
        """
        Take result from game and
        :param z:
        :return:
        """
        #z = torch.IntTensor(z)
        for mem in self._current_memory:

            p_turn = self._env.check_turn(mem.state)

            if z == -1:
                mem.z = 0.0001
            elif p_turn != z: # if winner != player of node
                mem.z = -1
            elif p_turn == z:
                mem.z = 1
            else:
                raise Exception("Incorrect winner passed")

            self._memory.push(mem)

        self.reset_current_memory()

    def train_network(self):

        if len(self._memory) < self._batch_size:
            return

        for _ in range(self._epochs):

            data = self._memory.sample(self._batch_size)

            b, p, v = zip(*[d for d in data])

            target_p = torch.FloatTensor(p)
            target_v = torch.FloatTensor(v)
            state = torch.FloatTensor(b)

            p, v = self._nn(state)
            loss_value = self.loss_v(target_v,v)

            loss_policy = self.loss_pi(target_p, p)

            loss = loss_value + loss_policy #loss_policy
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):

        #return torch.sum((targets - outputs) ** 2)
        return torch.sum((targets - outputs) ** 2) / targets.size()[0]


class DesignerZero(Designer):

    def __init__(self,env,agent_config,exp_config,_run=False):
        super().__init__(env,agent_config,exp_config, _run=_run)

        self._train_iters = exp_config["train_iters"]
        self.current_best = self.load_agent(self._agent1_config)
        self.current_player = copy.deepcopy(self.current_best)

    def play_game(self,render,agent1,agent2,iter=None,game_num=0):

        self.env.reset()

        curr_player = agent1

        game_array = []

        while True:

            action = curr_player.act(self.env.board)

            if self._record_all:
                curr_board = self.env.board.copy()
                b_hash = hash((curr_board.tobytes(),))
                self._run.info[f"action_iter={iter}_{b_hash}_{game_num}"] = (curr_board.tolist(),int(action))

            curr_state, action, next_state, r = self.env.step(action)

            if render:
                game_array.append(self.env.board.copy().tolist())
                self.env.render()

            if r != 0:
                if self._record_all:
                    self._run.info[f"game_{iter}_{game_num}_result"] = r
                break

            curr_player = agent2 if curr_player == agent1 else agent1

        return self.env.winner

    def run(self):
        """
            0: Self Play - Store memories
            1: Train network
            2: Evaluate - Decide current best player
            :return:
        """

        # Initialize run by training a current best agent
        # self.train(self.current_best,self._train_iters)

        vs_minimax = []
        vs_best = []
        for iter in range(self._iters):

            print(f"Running iteration {iter}")

            # Self Play
            self.train(self.current_player,self._train_iters)

            # Train Network
            self.current_player.train_network()

            #self.current_player._memory.save("")

            # For Evaluation the players should be taking max actions
            self.current_player._act_max = True
            self.current_best._act_max = True

            if (iter % 2) == 0:
                # Evaluate
                print(" ---------- Eval as player 1 vs minimax ---------")
                self.current_player.reset_tree()
                p1_result = self.run_eval(self.current_player, self.agent2,self._eval_iters,iter=iter)
                vs_minimax.append(p1_result)


            print("---------- Current Player vs Current Best ____________ ")
            self.current_player.reset_tree()
            self.current_best.reset_tree()

            curr_result = self.run_eval(self.current_player,self.current_best,1,iter=iter)

            self.current_player.reset_tree()
            self.current_best.reset_tree()

            curr_result2 = self.run_eval(self.current_best,self.current_player,1,iter=iter)

            tot_result = curr_result + -1*curr_result2

            vs_best.append(tot_result)

            if tot_result > 0:
                self.current_best = copy.deepcopy(self.current_player)
            else:
                self.current_player = copy.deepcopy(self.current_best)

            self.current_player.reset_tree()
            self.current_best.reset_tree()

            if self._run:
                self._run.log_scalar("tot_p1_wins",p1_result)
                self._run.log_scalar("currp_vs_currbest",tot_result)
                #self._run.log_scalar("tot_p2_wins",-1*p2_result)

            self.current_player._act_max = False

            self.current_player.reset_memory()
            self.current_best.reset_memory()

        plt.plot(vs_minimax,label="vs minimax")

        plt.plot(vs_best,label="vs best")

        plt.show()

    def run_eval(self,agent1,agent2,iters,render=False,iter=None):

        """
            This method evaluates the current agent
            :return:
        """

        agent1.player = 1
        agent2.player = 2

        result = 0
        for i in range(iters):
            winner = self.play_game(self._render, agent1, agent2, iter=iter,game_num=i)
            if winner == agent1.player:
                result += 1
            elif winner == agent2.player:
                result -= 1

        return result

    def train(self,agent,iters):

        agent.reset_current_memory()

        for _ in range(iters):

            agent.reset_tree()

            z = self.play_game(False,agent,agent)

            agent.store_memory(z)


