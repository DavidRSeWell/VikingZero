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

    data = None

    if config.split(".")[-1] == "yaml":
        print("Processing Yaml Config File")

        with open(config) as f:
            data = yaml.load(f, Loader=yaml.FullLoader)

    print("----------------PASSED PARAMS -----------------------")
    print(json.dumps(data, indent=4, sort_keys=True))

    env_name = data["env"]
    print(f"Loading environment {env_name}")
    env = load_env(env_name)()

    agent_config = data["agent_config"]

    exp_config = data["exp_config"]

    designer = None

    if "AlphaZero" in agent_config:
        print("Running AlphaGO Zero Experiment. Loading Appropriate designer")
        designer = DesignerZero
    else:
        designer = Designer

    designer = designer(env,agent_config,exp_config)

    print("Done loading configurations now running experiment")
    exp_logger = designer.run()

    print("Done running experiment")
    if exp_config["plot_data"]:
        exp_logger.plot_metrics()

    print("Showing results")



