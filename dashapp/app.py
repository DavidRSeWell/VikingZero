import dash
import dash_bootstrap_components as dbc

external_stylesheets = ['https://codepen.io/anon/pen/mardKv.css']

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

app.config.suppress_callback_exceptions = True
