import h5py
import numpy as np
import plotly.figure_factory as ff
import pandas as pd

import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_table

from abc import ABC, abstractmethod
from dash.dependencies import Output,Input

board_f = h5py.File("/Users/befeltingu/Documents/GitHub/VikingZero/notebooks/test2.hdf5", "r")
oracle_df = pd.read_hdf('/Users/befeltingu/Documents/GitHub/VikingZero/notebooks/oracle.hdf5',key='df')
oracle_df['id'] = oracle_df['key']
oracle_df.set_index('id', inplace=True, drop=False)


class Component(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def layout(self) -> html.Div:
        pass


class Connect4Board(Component):

    def __init__(self,board: np.array =None):

        super().__init__()

        self._board = board

    @property
    def board(self):
        return self._board

    @board.setter
    def board(self,board):
        self._board = board

    @property
    def layout(self):


        if len(self._board.shape) == 2:
            bs = self.display_board(self._board.flatten())
        else:
            bs = self.display_board(self._board)

        comp = dcc.Textarea(
                id='connect4_board',
                value=bs,
                style={'width': 300, 'height':300 },
                ),

        return comp

    @staticmethod
    def display_board(board):
        columns = 7
        rows = 6

        board = board.astype(np.int)

        def print_row(values, delim="|"):
            return f"{delim} " + f" {delim} ".join(str(v) for v in values) + f" {delim}\n"

        row_bar = "+" + "+".join(["---"] * columns) + "\n"
        out = row_bar
        for r in range(rows):
            out = out + \
                  print_row(board[r * columns: r * columns + columns]) + row_bar

        return out


class OracleTable(Component):

    def __init__(self):
        super().__init__()

    def layout(self):

        layout = html.Div([
            dash_table.DataTable(
                id='oracle-table',
                columns=[
                    {"name":i, "id":i ,"deletable": True, "selectable": True} for i in list(oracle_df.columns) if i != 'id'
                ],
                data=oracle_df.to_dict('records'),
                editable=True,
                filter_action="native",
                sort_action="native",
                sort_mode='multi',
                row_selectable='single',
                row_deletable=True,
                selected_rows=[],
                page_action='native',
                page_current=0,
                page_size=10,
            )
        ])

        return layout


