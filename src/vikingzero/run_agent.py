"""
Main entry point to the VikingZero Library
This point will consume a yaml file or some
configuration file (currently just yaml) and
process it and run the desired config
"""
import json
import yaml

from vikingzero.utils import load_env
from vikingzero.designer import DesignerZero,Designer


def run_agent(config):

    assert (type(config) == str) or (type(config) == dict)

    data = None

    if type(config) == str:
        if config.split(".")[-1] == "yaml":
            print("Processing Yaml Config File")

            with open(config) as f:
                data = yaml.load(f, Loader=yaml.FullLoader)

    elif type(config) == dict:
        data = config

    print("----------------PASSED PARAMS -----------------------")
    print(json.dumps(data, indent=4, sort_keys=True))

    env_name = data["env"]
    print(f"Loading environment {env_name}")
    env = load_env(env_name)()

    agent_config = data["agent_config"]

    exp_config = data["exp_config"]

    designer = Designer

    agent1 = agent_config["agent1"]["agent"]
    agent2 = agent_config["agent2"]["agent"]

    if "AlphaZero" in (agent1, agent2):
        print("Running AlphaGO Zero Experiment. Loading Appropriate designer")
        designer = DesignerZero

    designer = designer(env,agent_config,exp_config)

    print("Done loading configurations now running experiment")
    exp_logger,agent = designer.run()

    print("Done running experiment")
    if exp_config["plot_data"]:
        print("Showing results")
        exp_logger.plot_metrics()

    return exp_logger,agent


