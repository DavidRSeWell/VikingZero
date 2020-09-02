import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
import plotly.express as px


class SacredExp:

    def __init__(self,exp):
        self._exp = exp

        self.graph = self.load_graph()
        self.games = self.load_games()
        self.info = self.load_info()

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
        """ 
        input_groups = html.Div(
            [
                dbc.InputGroup(
                    [
                        dbc.InputGroupAddon("", addon_type="prepend"),
                        dbc.Input(placeholder="Username"),
                    ],
                    className="mb-3",
                ),
                dbc.InputGroup(
                    [
                        dbc.Input(placeholder="Recipient's username"),
                        dbc.InputGroupAddon("@example.com", addon_type="append"),
                    ],
                    className="mb-3",
                ),
                dbc.InputGroup(
                    [
                        dbc.InputGroupAddon("$", addon_type="prepend"),
                        dbc.Input(placeholder="Amount", type="number"),
                        dbc.InputGroupAddon(".00", addon_type="append"),
                    ],
                    className="mb-3",
                ),
                dbc.InputGroup(
                    [
                        dbc.InputGroupAddon("With textarea", addon_type="prepend"),
                        dbc.Textarea(),
                    ],
                    className="mb-3",
                ),
                dbc.InputGroup(
                    [
                        dbc.Select(
                            options=[
                                {"label": "Option 1", "value": 1},
                                {"label": "Option 2", "value": 2},
                            ]
                        ),
                        dbc.InputGroupAddon("With select", addon_type="append"),
                    ]
                ),
            ]
        )
        
        return input_groups
        """


