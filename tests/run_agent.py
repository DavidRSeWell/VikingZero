import os

from vikingzero.run_agent import run_agent

curr_dir = os.path.abspath(os.getcwd())
config = curr_dir + "/test_alphago.yaml"

run_agent(config)
