"""A standard machine learning task using sacred's magic."""
from sacred import Experiment
from sacred.observers import FileStorageObserver

from vikingzero.utils import load_env
from vikingzero.agents.tictactoe_agents import TicTacToeMCTS,TicTacToeMinMax
from vikingzero.designers.connect4_designer import Connect4Designer
from vikingzero.environments.tictactoe_env import TicTacToe

ex = Experiment("tictactoe")

ex.observers.append(FileStorageObserver("test_sacred"))

ex.add_config("test_sacred.yaml")


@ex.capture
def run_ex(env,iters,agent1,agent2,render):

    env = load_env(env)()

    designer = Connect4Designer(iters=iters,env=env,agent1_config=agent1,
                                agent2_config=agent2)

    designer.run(render=render)


@ex.main
def main():
    run_ex()


if __name__ == "__main__":
    ex.run_commandline()
