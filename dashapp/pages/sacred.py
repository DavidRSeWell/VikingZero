import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc

try:
    from ..components.connect4 import Connect4Board,OracleTable
except:
    from components.connect4 import Connect4Board,OracleTable


class SacredExpPage:

    def __init__(self,app):
        self._app = app

    @property
    def layout(self):

        comp = dbc.Container([
            html.H1("Board"),
            html.Hr(),
            dbc.Row(
                [

                ],
                align="center",
                id="connect4state_main"),
            dbc.Row(
                [
                    dbc.Button("test",id="fetch_board",color="primary")
                ]
            ),

            dbc.Row([self._oracle_table.layout()],id="oracle-table-row")
        ])

        return comp
