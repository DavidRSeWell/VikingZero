"""A standard machine learning task using sacred's magic."""
from sacred import Experiment
from sacred.observers import FileStorageObserver,MongoObserver

from vikingzero.utils import load_env
from vikingzero.agents.alphago import AlphaZero,DesignerZero

#TODO allow passing in command line args

#ex = Experiment("AlphaGoZero_connect4")
ex = Experiment("AlphaGoZero_tictactoe")

#ex.observers.append(MongoObserver(url="localhost:27017",db_name="VikingZero"))
ex.observers.append(FileStorageObserver("alphago"))

#ex.add_config("test_alphago.yaml")
ex.add_config("tictactoe_alphago.yaml")

#ex.captured_out_filter = lambda text: 'Output capturing turned off.'

@ex.capture
def run_ex(env,agent_config,exp_config,_run):

    env = load_env(env)()

    designer = DesignerZero(env,agent_config,exp_config,_run)

    designer.run()


@ex.main
def main():
    run_ex()


if __name__ == "__main__":
    ex.run_commandline()
