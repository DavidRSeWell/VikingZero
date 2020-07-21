from sacred import Experiment

from agents.connect4_agent import RandomConnect4Agent
from environments.connect4_env import Connect4

ex = Experiment('Connect4_Experiment')


@ex.config
def connect4_config():

    env = Connect4()

    iters = 100

    agent1_config = {
        'agent': RandomConnect4Agent
    }

    agent2_config = {
        'agent': RandomConnect4Agent
    }

    render = False


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
        agent = agent_config['agent'](self.env)

        return agent

    def play_game(self,render):

        self.env.reset()

        curr_player = self.agent1

        while True:

            action = curr_player.act(self.env.board)

            curr_state, action, next_state, winner = self.env.step(action)

            if render:
                self.env.render()

            if winner != 0:
                break

            curr_player = self.agent2

        return winner

    def run(self,render=False):

        winners = {
            '1':0, # player1
            '-1':0, # player2
            '2':0 # draw
        }
        for iter in range(self._iters):

            winner = self.play_game(render)
            winners[str(int(winner))] += 1

        print(winners)


@ex.capture
def run_ex(env,iters,agent1_config,agent2_config,render):

    designer = Connect4Designer(iters=iters,env=env,agent1_config=agent1_config,
                                agent2_config=agent2_config)

    designer.run(render=render)


@ex.main
def main():

    run_ex()


if __name__ == "__main__":

    ex.run_commandline()


