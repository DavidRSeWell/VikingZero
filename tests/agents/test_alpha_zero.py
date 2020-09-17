from vikingzero.agents.alphago import DesignerZero, AlphaZero
from vikingzero.agents.tictactoe_agents import TicTacMCTSNode,TicTacMiniNode
from vikingzero.environments.tictactoe_env import TicTacToe



def test_alphago_zero():

    agent_config = {

        "agent1":{
            "agent": "AlphaZero",
            #"node": TicTacMCTSNode,
            "n_sim":15,
            "batch_size": 32,
            "c": 1.41,
            "lr": 0.001,
            "max_mem_size": 250,
            "player":1
        },

        "agent2": {

            "agent": "TicTacToeMinMax",
            #"node": TicTacMiniNode,
            #"n_sim": 50,
            #"batch_size": 5,
            #"c": 1.41,
            "player": 2,
            "type": "minimax"
        },

    }

    exp_config = {
        "episodes": 30,
        "record_all": False,
        "record_every": 1,
        "eval_iters": 1,
        "render":True,
        "train_iters": 10
    }

    designer = DesignerZero(TicTacToe(),agent_config,exp_config,_run=False)

    designer.run()


if __name__ == "__main__":

    test_alphago_zero()
