"""A standard machine learning task using sacred's magic."""
from sacred import Experiment
from sacred.observers import MongoObserver

from vikingzero.utils import load_env
from vikingzero.designers.connect4_designer import Designer

ex = Experiment("tictactoe")

ex.observers.append(MongoObserver(url="localhost:27017",db_name="VikingZero"))

ex.add_config("test_tictactoe.yaml")


@ex.capture
def run_ex(env,agent_config,exp_config,_run):

    env = load_env(env)()

    designer = Designer(env,agent_config,exp_config,_run)

    designer.run()


@ex.main
def main():
    run_ex()


if __name__ == "__main__":
    ex.run_commandline()
