from vikingzero.agents.alphago import DesignerZero, AlphaZero
from vikingzero.agents.tictactoe_agents import TicTacMCTSNode
from vikingzero.environments.tictactoe_env import TicTacToe



def test_alphago_zero():

    agent_config = {

        "agent1":{

            "agent": "AlphaZero",
            "node": TicTacMCTSNode,
            "n_sim":50,
            "batch_size": 5,
            "c": 1.41,
            "player":1
        },

        "agent2": {

            "agent": "AlphaZero",
            "node": TicTacMCTSNode,
            "n_sim": 50,
            "batch_size": 5,
            "c": 1.41,
            "player": 1
        },

    }

    exp_config = {
        "episodes": 100,
        "record_all": False,
        "record_every": 1,
        "eval_iters": 1,
        "render":False,
        "train_iters": 2
    }

    designer = DesignerZero(TicTacToe(),agent_config,exp_config,_run=False)

    designer.run()


if __name__ == "__main__":

    test_alphago_zero()
