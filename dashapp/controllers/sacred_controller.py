

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
        """ 
        last_exp = exps[0]

        last_exp = dbc.ListGroupItem(str(last_exp),id={"type": "exp_item",
                         "index" : last_exp.id
                        },key="hey",active=True)

        items[0] = last_exp
        """

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
            Output("exp_panel","children"),[Input("exp-tabs","value"),Input("exp_id","data"),Input("game_info","data")]
        )
        def display_exp_tab(tab,data,game_data):

            if not self._current_exp:
                return []

            if tab == 'tab-info':
                return self._current_exp.info

            elif tab == 'tab-plots':
                return self._current_exp.graph

            elif tab == 'tab-games':
                if game_data is None:
                    return self._current_exp.game_panel(game_id=0, move_id=0)
                else:
                    return self._current_exp.game_panel(game_id=game_data["game_id"],move_id=game_data["move_id"])

        @app.callback(
            [Output("game_info","data")],[Input({"type":"game_num","index":ALL},"n_clicks"),
                                          Input("game_next","n_clicks")],[State("game_info","data")]
        )
        def change_game(game_index,game_next,game_info):


            ctx = dash.callback_context

            if not ctx.triggered:
                return [{"game_id": 0, "move_id": 0}]
            else:
                print("change game")
                prop_id = ctx.triggered[0]["prop_id"]
                #prop_id = json.load(ctx.triggered[0])["prop_id"]
                if "game_num" in prop_id:
                    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
                    button_id = json.loads(button_id)["index"]
                    return [{"game_id":button_id, "move_id":0}]

                elif "game_next" in prop_id:
                    print("game next")
                    game_info = game_info.copy()
                    game_id = game_info["game_id"]
                    current_game = self._current_exp.games[f"game_{game_id}"]
                    current_game_len = 4
                    game_info["move_id"] = min(game_info["move_id"] + 1, self._current_exp.games)
                    return [game_info]




        """
                @app.callback(
                    Output({"type": "exp_item","index":MATCH},"children"),[Input({"type":"exp_item","index": MATCH},"n_clicks")]
                )
                def active_value(index):
                    print("active value callback")
                    print(index)

                    if index:
                        print("not none")
                        exp = self.exps[index]
                        return dbc.ListGroupItem(str(exp), id={"type": "exp_item",
                                                         "index": index
                                                         }, key="hey", active=True)
        """
