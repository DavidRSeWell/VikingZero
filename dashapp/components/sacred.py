import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
import plotly.express as px


from .connect4 import Connect4Board


class SacredExp:

    def __init__(self,exp):
        self._exp = exp

        self.graph = self.load_graph()
        self.games = self.load_games()
        self.info = self.load_info()

    def game_panel(self,game_id=0,move_id=0):
        game = self._exp.games[f"game_{game_id}"]
        game = np.array(game)
        board = game[move_id]
        if self._exp.env.lower() == "tictactoe":
            board = board.reshape((3,3))

        board_comp = Connect4Board(board)

        panel = dbc.Row([

                dbc.Col(self.games,width=2),
                dbc.Col([html.Div(html.H2(f"Game={game_id} MoveID = {move_id}")),
                     html.Div(board_comp.layout),dbc.Button("Prev","game_prev"),
                     dbc.Button("Next",id="game_next")],width=10)
            ]
        )

        return panel

    def load_graph(self):

        y = self._exp.loss_data
        x = [i for i in range(len(y))]
        data = np.array([x,y]).T
        print(data.shape)
        df = pd.DataFrame(data,columns=["x","y"])
        fig = px.line(df,"x","y")

        return dcc.Graph(
            id='basic-interactions',
            figure=fig
        ),

    def load_games(self):

        game_list = self._exp.games

        keys = game_list.keys()

        items = [dbc.ListGroupItem(

            game, id={"type": "game_num",
                          "index": game.split("_")[-1]
                          }, key=game.split("_")[-1],
            action=True

        ) for game in keys]

        list_group = dbc.ListGroup(
           items
        )

        return list_group

    def load_info(self):

        config_info = self._exp.config
        agent_config = config_info["agent_config"]

        return html.Div(str(config_info))




