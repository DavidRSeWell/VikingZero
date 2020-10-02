import neptune
import yaml

from vikingzero.utils import load_env
from vikingzero.agents.alphago import DesignerZero

NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiYThhM2FkNmItNGJmOC00NTc3LWI1ZDctNmE1NTYxMDBmMzkyIn0="

neptune.init('befeltingu/sandbox',api_token=NEPTUNE_API_TOKEN)

yaml_file = "tictactoe_alphago.yaml"
#yaml_file = "test_alphago.yaml"

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
