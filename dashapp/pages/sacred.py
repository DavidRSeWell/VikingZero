import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_core_components as dcc

try:
    from ..controllers.sacred_controller import SacredController
    from ..components.sacred import SacredExp
except:
    from controllers.sacred_controller import SacredController
    from components.sacred import SacredExp


tabs_styles = {
    'height': '44px'
}

tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'fontWeight': 'bold'
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#119DFF',
    'color': 'white',
    'padding': '6px'
}


class SacredExpPage:

    def __init__(self,app,db):
        self._app = app
        self._db = db
        self._current_exp = None
        self.controller = SacredController(self._db)
        self.controller.register_callbacks(self._app)
        self._exp = self.controller.exps_list_group

    @property
    def layout(self):

        comp = html.Div([
            dcc.Store("exp_id"),
            dcc.Store("game_info"),
            html.H1("Experiments"),
            html.Hr(),
            dbc.Row([
                dbc.Col(self._exp,width=2,id="exp_list"),
                dbc.Col([
                      html.Div([
                        dcc.Tabs(id="exp-tabs", value='tab-info', children=[
                            dcc.Tab(label='info', value='tab-info', style=tab_style, selected_style=tab_selected_style),
                            dcc.Tab(label='plots', value='tab-plots', style=tab_style, selected_style=tab_selected_style),
                            dcc.Tab(label='games', value='tab-games', style=tab_style, selected_style=tab_selected_style),
                        ], style=tabs_styles)]
                    ),
                    html.Div(
                        [],id="exp_panel")
                ],
                width=10)
            ])
        ])
        return comp


