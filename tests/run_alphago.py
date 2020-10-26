import neptune
import yaml

from vikingzero.utils import load_env
from vikingzero.agents.alphago import DesignerZero
#from vikingzero.designer import DesignerZero

NEPTUNE_API_TOKEN=""

neptune_username = ""

neptune.init(f"{neptune_username}/sandbox",api_token=NEPTUNE_API_TOKEN)

yaml_file = "test_alphago.yaml"

# Load experiment variables
with open(yaml_file) as f:
    data = yaml.load(f,Loader=yaml.FullLoader)

neptune.create_experiment(name='AlphaGo',params=data)

def main():

    env = load_env(data["env"])()

    designer = DesignerZero(env,data["agent_config"],data["exp_config"],None)

    designer.run()


if __name__ == "__main__":
    main()
