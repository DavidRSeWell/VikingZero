import h5py
import pandas as pd
import plotly.figure_factory as ff
import numpy as np
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc

from abc import ABC, abstractmethod
from dash.dependencies import Output,Input

try:
    from ..components.connect4 import Connect4Board
except:
    from components.connect4 import Connect4Board


board_f = h5py.File("/Users/befeltingu/Documents/GitHub/VikingZero/notebooks/test2.hdf5", "r")
oracle_df = pd.read_hdf('/Users/befeltingu/Documents/GitHub/VikingZero/notebooks/oracle.hdf5',key='df')
oracle_df['id'] = oracle_df['key']
oracle_df.set_index('id', inplace=True, drop=False)


class Controller:

    def __init__(self):
        pass

    def register_callbacks(self,app):
        """

        :param app:
        :return:
        """
        '''
        @app.callback(Output("connect4state_main", "children"), [Input("fetch_board", "n_clicks")])
        def fetch_board(n_clicks):

            print("Fetch baord")
            # fetch board
            board = np.array([
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 2, 0, 0, 0],
                [0, 0, 2, 1, 0, 0, 0],
            ])

            connect4_baord = Connect4Board(board)

            return connect4_baord.layout
        '''

        @app.callback(
            Output('connect4state_main', 'children'),
            [Input('oracle-table', 'derived_virtual_row_ids'),
             Input('oracle-table', 'selected_row_ids'),
             Input('oracle-table', 'rows'),
             Input('oracle-table', 'selected_row_indices')])
        def update_graphs(row_ids, selected_row_ids, active_cell,selected_cells):

            print("update_graphs")
            #print(active_cell)
            print(selected_row_ids)
            print(selected_cells)
            if selected_row_ids:
                print(len(row_ids))
                #row = oracle_df.iloc[active_cell['row']]
                #key = row['key']
                key = selected_row_ids[0]
                print("key")
                print(key)
                board = board_f[key][...]

                board_ui = Connect4Board(board)

                return board_ui.layout


