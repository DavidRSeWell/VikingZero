import copy
import time
import matplotlib.pyplot as plt

from collections.abc import Iterable
from datetime import datetime
from tqdm import tqdm

# code in place if someone wants to use neptune to track experiments
try:
    import neptune
except:
    pass

from .utils import load_agent


class ExpLogger:
    """
    Class for containing functionality
    dealing with the handling of experiment data
    during and after an experiment is run. Acts as a go
    between for the designer and experiment data.
    """

    def __init__(self,exp_config,agent_config):
        self._agent_config = agent_config
        self._exp_config = exp_config
        self._exp_metrics = {"time": datetime.now()}
        self._run = None
        self.load_logger()

    def init_neptune(self):
        print("INIT NEPTUNE")
        neptune_api_token = self._exp_config["neptune_api_token"]
        neptune_name = self._exp_config["neptune_name"]
        exp_name = self._exp_config["exp_name"]

        data = {
            "agent_config": self._agent_config,
            "exp_config": self._exp_config
        }

        neptune.init(neptune_name,api_token=neptune_api_token)

        neptune.create_experiment(exp_name,params=data)

        self.exp_id = neptune.get_experiment().id

        return neptune.get_experiment()

    def init_tensorboard(self):
        from tensorboardX import SummaryWriter

        writer = SummaryWriter()

        data = {
            "agent_config": self._agent_config,
            "exp_config": self._exp_config
        }

        writer.add_hparams(self._exp_config,{})
        writer.add_hparams(self._agent_config,{})

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

        if type(logger_type) == str:
            if logger_type == "neptune":
                self.init_neptune()
                self._run = neptune.get_experiment()

            elif logger_type == "tensorboard":
                self._run = self.init_tensorboard()

            else:
                raise Exception(f"Unknown logger type {logger_type} given")

        else:
            print(f"Running without logger {logger_type} passed")
            return

    def log_metric(self,key,value):
        self._exp_metrics[key].append(value)

    def log_metrics(self,iter,iter_metrics):

        for k , v in iter_metrics.items():
            if k not in self._exp_metrics:
                self._exp_metrics[k] = []

        logger_type = self._exp_config["logger_type"]

        if logger_type == "neptune":
            self.log_neptune_metrics(iter_metrics)

        elif logger_type == "tensorboard":
            self.log_tensorboard_metrics(iter,iter_metrics)

        elif logger_type == "both":
            self.log_neptune_metrics(iter_metrics)
            self.log_tensorboard_metrics(iter,iter_metrics)

        # Log to self iter_metric regardless of logger type
        for k, v in iter_metrics.items():
            self.log_metric(k,v)

        return None

    def log_neptune_metrics(self,iter_metrics):
        for key , value in iter_metrics.items():
            if value or value == 0:
                neptune.log_metric(key,value)

    def log_tensorboard_metrics(self,iter,iter_metrics):

        for key , value in iter_metrics.items():
            if value or value == 0:
                self._run.add_scalar(key,value,iter)

    def plot_metrics(self):
        for k,v in self._exp_metrics.items():
            if not isinstance(v,Iterable):
                continue
            v = [x for x in v if x is not None]
            plt.plot(v)
            plt.title(k)
            plt.legend()
            plt.show()


class Designer:

    def __init__(self,env,agent_config,exp_config):

        self._agent1_config = agent_config["agent1"]
        self._agent2_config = agent_config["agent2"]
        self._eval_iters = exp_config["eval_iters"]
        self._iters = exp_config["episodes"]
        self._record_all = exp_config["record_all"]
        self._record_every = exp_config["record_every"]
        self._render = exp_config["render"]
        self._train_iters = exp_config["train_iters"]

        self.env = env
        self.exp_logger = ExpLogger(exp_config,agent_config)
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
        :return: ExpLogger - Metrics logged during run
        """

        for iter in tqdm(range(self._iters)):
            print(f"Running iteration {iter}")
            iter_metrics = {
                "tot_p1_wins": None,
                "tot_p2_wins": None
            }

            if (iter % self._record_every) == 0 or (iter == self._iters - 1):

                p1_r = self.run_eval(self.agent1,self.agent2,self._eval_iters,render=self._render,iter=iter)
                p2_r = self.run_eval(self.agent2,self.agent1,self._eval_iters,render=self._render,iter=iter)

                iter_metrics["tot_p1_wins"] = p1_r
                iter_metrics["tot_p2_wins"] = -1*p2_r

                self.exp_logger.log_metrics(iter,iter_metrics)

            self.train(self.agent1,self._train_iters)

        return self.exp_logger, self.agent1

    def run_eval(self,agent1,agent2,iters,render=False,iter=None):
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

    def train(self,agent,iters):

         for _ in range(iters):
             self.play_game(False,agent,agent)


class DesignerZero(Designer):

    def __init__(self,env,agent_config,exp_config):
        super().__init__(env,agent_config,exp_config)

        self._agent_config = agent_config
        self._exp_config = exp_config
        self._run_evaluator = exp_config["run_evaluator"]
        self._run_minimax_eval = exp_config["run_minimax_eval"]
        self._save_model = exp_config["save_model"]
        self._train_iters = exp_config["train_iters"]
        self.current_best = self.load_agent(self._agent1_config)
        self.current_player = copy.deepcopy(self.current_best)
        self.eval_threshold = exp_config["eval_threshold"]
        self.exp_id = None

        ###########
        # LOSS
        ###########
        self.avg_loss = []
        self.avg_value_loss = []
        self.avg_policy_loss = []

    def compute_metrics(self,iter_metrics) -> dict:

        # Evaluate
        if self._render:
            print(" ---------- Eval as player 1 vs minimax ---------")
        p1_result = self.run_eval(self.current_player, self.agent2,self._eval_iters,iter=iter)

        if self._render:
            print(" ---------- Eval as player 2 vs minimax ---------")
        p2_result = self.run_eval(self.agent2, self.current_player, self._eval_iters, iter=iter)
        p2_result *= -1

        iter_metrics["tot_p1_wins"] = p1_result
        iter_metrics["tot_p2_wins"] = p2_result

        if self._run_minimax_eval:
            policy_correct, mcts_correct, mcts_bar_correct = self.current_player.loss_minimax()
            iter_metrics["policy_minimax_score"] = policy_correct
            iter_metrics["mcts_minimax_score"] = mcts_correct
            #iter_metrics["mcts_bar_minimax_score"] = mcts_bar_correct

        return iter_metrics

    def play_game(self,render,agent1,agent2,iter=None,game_num=0):

        self.env.reset()

        if hasattr(agent1,"reset"):
            agent1.reset()
        if hasattr(agent2,"reset"):
            agent2.reset()

        curr_player = agent1

        game_array = []

        last_action = None

        while True:

            action = curr_player.act(self.env.board)
            if type(action) == tuple:
                action , p_a = action

            if self._record_all:
                curr_board = self.env.board.copy()
                b_hash = hash((curr_board.tobytes(),))
                #self._run.info[f"action_iter={iter}_{b_hash}_{game_num}"] = (curr_board.tolist(),int(action))

            curr_state, action, next_state, r = self.env.step(action)

            if hasattr(agent1,"update_state"):
                agent1.update_state(curr_state,next_state)

            if hasattr(agent2,"update_state"):
                agent2.update_state(curr_state,next_state)

            if render and hasattr(curr_player, "display_state_info"):
                curr_player.display_state_info(curr_state,last_action)
                print(f"Agent chose action ={action}")

            if render:
                game_array.append(self.env.board.copy().tolist())
                self.env.render()

            if r != 0:
                break

            curr_player = agent2 if curr_player == agent1 else agent1

            last_action = action

        return self.env.winner

    def run(self):
        """
            0: Self Play - Store memories
            1: Train network
            2: Evaluate - Decide current best player
            :return:
        """

        # Initialize run by training a current best agent

        iter_metrics = self.compute_metrics({})
        self.exp_logger.log_metrics(0, iter_metrics)

        for iter in tqdm(range(self._iters)):

            print(f"Running iteration {iter}")

            iter_metrics = {
                "Total Loss":None,
                "Value Loss":None,
                "Policy Loss":None,
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

            iter_metrics["Total Loss"] = float(avg_total)
            iter_metrics["Policy Loss"] = float(avg_policy)
            iter_metrics["Value Loss"] = float(avg_val)


            e_time = time.time()
            min_time = (e_time - s_time) / 60.0
            print(f"Training network time ={min_time}")

            if (iter % self._record_every) == 0:

                iter_metrics = self.compute_metrics(iter_metrics)

            if self._run_evaluator:
                print("---------- Current Player vs Current Best ____________ ")

                curr_result = self.run_eval(self.current_player,self.current_best,10,iter=iter)

                curr_result2 = self.run_eval(self.current_best,self.current_player,10,iter=iter)

                tot_result = curr_result + -1*curr_result2

                if (tot_result >= self.eval_threshold):
                    print(f"Changing Agent on iteration = {iter}")
                    self.current_best = copy.deepcopy(self.current_player)

                else:
                    self.current_player = copy.deepcopy(self.current_best)

            # Record metrics each iteration
            self.exp_logger.log_metrics(iter + 1,iter_metrics)

            if self._save_model:
                if self._run_evaluator:
                    self.current_best.save(self.exp_id)
                else:
                    self.current_player.save(self.exp_id)

        return self.exp_logger,self.current_player

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
