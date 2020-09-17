import numpy as np

from sacred import Experiment


try:
    from ..utils import load_agent,load_env
    from ..agents.connect4_agent import RandomConnect4Agent, Connect4MCTS
    from ..environments.connect4_env import Connect4
except:
    from vikingzero.utils import load_agent,load_env
    from vikingzero.agents.connect4_agent import RandomConnect4Agent, Connect4MCTS
    from vikingzero.environments.connect4_env import Connect4


class Connect4Designer:

    def __init__(self,iters,env,agent1_config,agent2_config):


        self._agent1_config = agent1_config
        self._agent2_config = agent2_config
        self._iters = iters

        self.env = env
        self.agent1 = self.load_agent(agent1_config)
        self.agent2 = self.load_agent(agent2_config)

    def load_agent(self,agent_config):
        """
        Instantiate an agent from its configuration
        :param agent_config: dict
        :return: class
        """
        agent_config = agent_config.copy()
        agent_name = agent_config["agent"]
        del agent_config["agent"]
        Agent = load_agent(agent_name)
        agent = Agent(self.env, **agent_config)
        return agent

    def play_game(self,render):

        self.env.reset()

        curr_player = self.agent1

        while True:

            action = curr_player.act(self.env.board)

            curr_state, action, next_state, r = self.env.step(action)

            if render:
                self.env.render()

            if r != 0:
                break

            curr_player = self.agent2 if curr_player == self.agent1 else self.agent1

        return self.env.winner

    def run(self,render=False,show_every=5):

        winners = {
            "1":0, # player1
            "2":0, # player2
            "-1":0 # draw
        }

        for iter in range(self._iters):
            print(f"Running iteration {iter}")

            if (iter % show_every) == 0:
                winner = self.play_game(render)

            else:
                winner = self.play_game(render=False)

            winners[str(int(winner))] += 1

        print(winners)


class Designer:

    def __init__(self,env,agent_config,exp_config,_run=False):

        self._agent1_config = agent_config["agent1"]
        self._agent2_config = agent_config["agent2"]
        self._eval_iters = exp_config["eval_iters"]
        self._iters = exp_config["episodes"]
        self._record_all = exp_config["record_all"]
        self._record_every = exp_config["record_every"]
        self._render = exp_config["render"]
        self._run = _run # Comes from sacred library to track the run

        self.env = env
        self.agent1 = self.load_agent(self._agent1_config)
        self.agent2 = self.load_agent(self._agent2_config)

    def load_agent(self,agent_config):
        """
        Instantiate an agent from its configuration
        :param agent_config: dict
        :return: class
        """
        agent_config = agent_config.copy()
        agent_name = agent_config["agent"]
        del agent_config["agent"]
        Agent = load_agent(agent_name)
        agent = Agent(self.env, **agent_config)
        return agent

    def play_game(self,render,agent1,agent2,iter=None):

        self.env.reset()

        curr_player = agent1

        game_array = []

        while True:

            action = curr_player.act(self.env.board)

            if self._record_all:
                curr_board = self.env.board.copy()
                b_hash = hash((curr_board.tobytes(),))
                self._run.info[f"action_iter={iter}_{b_hash}"] = (curr_board.tolist(),int(action))

            curr_state, action, next_state, r = self.env.step(action)

            if render:
                game_array.append(self.env.board.copy().tolist())
                self.env.render()

            if r != 0:
                if self._record_all:
                    self._run.info[f"game_{iter}_result"] = r
                break

            curr_player = agent2 if curr_player == agent1 else agent1

        if render:
            self._run.info[f"game_{iter}"] = game_array
            pass

        return self.env.winner

    def run(self):
        """
        :param _run: Used in conjunction with sacred for recording tests
        :return:
        """

        for iter in range(self._iters):
            print(f"Running iteration {iter}")

            if (iter % self._record_every) == 0 or (iter == self._iters - 1):

                r = self.run_eval(iter=iter)

                self._run.log_scalar(r"tot_wins",r)

            #TODO Allow for self-play as a setting
            self.play_game(False,self.agent1,self.agent2,iter)

    def run_eval(self,iter=None):
        """
        This method evaluates the current agent
        :return:
        """
        result = 0
        for i in range(self._eval_iters):
            if i == 0:
                winner = self.play_game(self._render,self.agent1,self.agent2,iter=iter)
            else:
                winner = self.play_game(self._render,self.agent1,self.agent2)

            if winner == 2:
                result -= 1
            elif winner == 1:
                result += 1

        return result

    def train(self,iters):

         for _ in range(iters):
             self.play_game(self._render,self.agent1,self.agent1)


