import copy
from datetime import time

import neptune

from .utils import load_agent, load_env

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
                #self._run.info[f"action_iter={iter}_{b_hash}"] = (curr_board.tolist(),int(action))

            curr_state, action, next_state, r = self.env.step(action)

            if render:
                game_array.append(self.env.board.copy().tolist())
                self.env.render()

            if r != 0:
                if self._record_all:
                    #self._run.info[f"game_{iter}_result"] = r
                    pass
                break

            curr_player = agent2 if curr_player == agent1 else agent1

        if render:
            #self._run.info[f"game_{iter}"] = game_array
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

class DesignerZero(Designer):

    def __init__(self,env,agent_config,exp_config,_run=False, eval_threshold = 1):
        super().__init__(env,agent_config,exp_config, _run=_run)

        self._agent_config = agent_config
        self._exp_config = exp_config
        self._run_evaluator = exp_config["run_evaluator"]
        self._save_model = exp_config["save_model"]
        self._train_iters = exp_config["train_iters"]
        self.current_best = self.load_agent(self._agent1_config)
        self.current_player = copy.deepcopy(self.current_best)
        self.eval_threshold = eval_threshold
        self.exp_id = self.load_exp_id()

        self.load_logger()
        ###########
        # LOSS
        ###########
        self.avg_loss = []
        self.avg_value_loss = []
        self.avg_policy_loss = []

    def init_neptune(self):
        neptune_api_token = self._exp_config["neptune_api_token"]
        neptune_name = self._exp_config["neptune_name"]
        exp_name = self._exp_config["exp_name"]

        data = {
            "agent_config": self._agent_config,
            "exp_config": self._exp_config
        }

        neptune.init(neptune_name,api_token=neptune_api_token)

        neptune.create_experiment(exp_name,params=data)

    def init_tensorboard(self):
        from torch.utils.tensorboard import SummaryWriter

        exp_name = self._exp_config["exp_name"]
        # default `log_dir` is "runs" - we'll be more specific here
        writer = SummaryWriter(f'runs/{exp_name}')

        return writer

    def load_exp_id(self):
        if self._run:
            return self._run.id
        else:
            try:
                return neptune.get_experiment().id
            except:
                return None

    def load_logger(self):

        logger_type = self._exp_config["logger_type"]
        if logger_type:
            if logger_type == "neptune":
                self.init_neptune()
                self._run = neptune.get_experiment()

            elif logger_type == "tensorboard":
                self._run = self.init_tensorboard()

        else:
            return

    def log_metrics(self,iter):

        logger_type = self._exp_config["logger_type"]

        if logger_type == "neptune":
            self.log_neptune_metrics(iter)

        elif logger_type == "tensorboard":
            self.log_tensorboard_metrics(iter)

        elif logger_type == "both":
            self.log_neptune_metrics(iter)
            self.log_tensorboard_metrics(iter)

        return

    def log_neptune_metrics(self,iter):
        pass

    def log_tensorboard_metrics(self,iter):
        pass

    def play_game(self,render,agent1,agent2,iter=None,game_num=0):

        self.env.reset()

        try:
            agent1.reset()
        except:
            pass
        try:
            agent2.reset()
        except:
            pass

        curr_player = agent1

        game_array = []

        while True:

            action = curr_player.act(self.env.board)
            if type(action) == tuple:
                action , p_a = action

            if self._record_all:
                curr_board = self.env.board.copy()
                b_hash = hash((curr_board.tobytes(),))
                #self._run.info[f"action_iter={iter}_{b_hash}_{game_num}"] = (curr_board.tolist(),int(action))

            curr_state, action, next_state, r = self.env.step(action)

            if render:
                game_array.append(self.env.board.copy().tolist())
                self.env.render()

            if r != 0:
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

        vs_minimax = []
        vs_best = []
        for iter in range(self._iters):

            print(f"Running iteration {iter}")

            iter_metrics = {
                "train_avg_loss":None,
                "train_val_loss":None,
                "train_policy_loss":None,
                "tot_p1_wins":None,
                "tot_p2_wins":None,


            }

            # Self Play
            s_time = time.time()
            self.train(self.current_player,self._train_iters)
            e_time = time.time()
            min_time = (e_time - s_time) / 60.0
            print(f"Training time ={min_time} For {self._train_iters} iters")

            # Train Network
            s_time = time.time()
            avg_total, avg_policy, avg_val = self.current_player.train_network()

            e_time = time.time()
            min_time = (e_time - s_time) / 60.0
            print(f"Training network time ={min_time}")

            if (iter % self._record_every) == 0:
                # Evaluate
                print(" ---------- Eval as player 1 vs minimax ---------")
                p1_result = self.run_eval(self.current_player, self.agent2,self._eval_iters,iter=iter)
                vs_minimax.append(p1_result)

                print(" ---------- Eval as player 2 vs minimax ---------")
                p2_result = self.run_eval(self.agent2, self.current_player, self._eval_iters, iter=iter)
                p2_result *= -1
                vs_minimax.append(p2_result)

            print("---------- Current Player vs Current Best ____________ ")

            if self._run_evaluator:
                curr_result = self.run_eval(self.current_player,self.current_best,10,iter=iter)

                curr_result2 = self.run_eval(self.current_best,self.current_player,10,iter=iter)

                tot_result = curr_result + -1*curr_result2

                vs_best.append(tot_result)

                if (tot_result >= self.eval_threshold):
                    print(f"Changing Agent on iteration = {iter}")
                    self.current_best = copy.deepcopy(self.current_player)

                else:
                    self.current_player = copy.deepcopy(self.current_best)

            # Record metrics each iteration
            self.log_metrics(iter,iter_metrics)

            if self._save_model:
                self.current_best.save()


    def run_eval(self,agent1,agent2,iters,render=False,iter=None):

        """
            This method evaluates the current agent
            :return:
        """

        agent1.player = 1
        agent2.player = 2

        try:
            agent1.eval()
        except:
            pass
        try:
            agent2.eval()
        except:
            pass
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

        agent.train()

        for _ in range(iters):

            z = self.play_game(False,agent,agent)

            agent.store_memory(z)
