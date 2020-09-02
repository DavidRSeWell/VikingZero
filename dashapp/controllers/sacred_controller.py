

import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import json

from dash.dependencies import Output,Input,State,ALL,MATCH

try:
    from ..components.sacred import SacredExp
except:
    from components.sacred import SacredExp


class SacredController:

    def __init__(self,db):
        self._current_exp = None
        self._db = db
        self.exps = self._db.load_all()
        self.exps_list_group = self.load_experiments()

    def load_experiments(self):

        exps = self._db.load_all()

        items = [dbc.ListGroupItem(

            str(exp),id={"type": "exp_item",
                         "index" : exp.id
                        },key=f"{exp.id}",
            action=True

        ) for exp in exps]

        # set last one to active
        last_exp = exps[0]

        last_exp = dbc.ListGroupItem(str(last_exp),id={"type": "exp_item",
                         "index" : last_exp.id
                        },key="hey",active=True)

        items[0] = last_exp

        list_group = dbc.ListGroup(
           items
        )

        # initialize initial graph for the most recent experiment
        return list_group

    def register_callbacks(self,app):


        @app.callback(
            Output("exp_id","data"),[Input({'type': 'exp_item', 'index': ALL},"n_clicks")]

        )
        def display_output(index):
            ctx = dash.callback_context
            if not ctx.triggered:
               button_id = 'No clicks yet'
            else:
                button_id = ctx.triggered[0]['prop_id'].split('.')[0]
                button_id = json.loads(button_id)["index"]
                exp = self._db.load(button_id)
                exp_comp = SacredExp(exp)
                self._current_exp = exp_comp
                return exp.id

        @app.callback(
            Output("exp_panel","children"),[Input("exp-tabs","value"),Input("exp_id","data")]
        )
        def display_exp_tab(tab,data):

            if not self._current_exp:
                return []

            if tab == 'tab-info':
                return self._current_exp.info

            elif tab == 'tab-plots':
                return self._current_exp.graph

            elif tab == 'tab-games':
                return self._current_exp.games
