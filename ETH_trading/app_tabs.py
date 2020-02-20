import pandas as pd
import numpy as np
import dash_core_components as dcc
import dash_html_components as html
import dash_table


""" Some constants to construct the tabs"""


diary = 'trading_diary.xlsx'
# TODO: I do not like that I have to define this variable here

# DICTIONARIES FOR DROPDOWNS AND RABIO BUTTONS ETC.:
pairs = [
    {'label': 'ETH/USDT', 'value': 'ETHUSDT'},
    {'label': 'BTC/USDT', 'value': 'BTCUSDT'},
    {'label': 'XRP/USDT', 'value': 'XRPUSDT'},
         ]

types = [
    {'label': 'pullback to value', 'value': 'pullback to value'},
    {'label': 'ATR extreme', 'value': 'ATR extreme'},
    {'label': 'price rejection', 'value': 'price rejection'}  # support/resistance
]

directions = [
    {'label': 'LONG', 'value': 'LONG'},
    {'label': 'SHORT', 'value': 'SHORT'}
]

# TABLE LABELS FOR OPEN AND CLOSED TRADES:
open_trade_cols = ['pair', 'size', 'entry', 'stop', 'direction']
open_trade_dict = [{'name': c, 'id': c} for c in open_trade_cols]

closed_trade_cols = ['pair', 'size', 'entry', 'exit', 'stop', 'P/L (USDT)', 'P/L (%)', 'pre capital',
                     'risk', 'rrr', 'direction', 'type', 'confidence', 'note']
closed_trade_dict = [{'name': c, 'id': c} for c in closed_trade_cols]


""" Functions to read trades from the records """


def get_open_trades(record_file):
    open_trades = pd.read_excel(record_file, sheet_name='open')
    open_trades = open_trades.drop(columns=['date'])
    table_data = open_trades.to_dict(orient='records')
    return table_data


def get_closed_trades(record_file):
    closed_trades = pd.read_excel(record_file, sheet_name='closed')
    # closed_trades = closed_trades.drop(columns=['date'])
    # TODO: make sure to drop the right columns here to match the table dict!
    table_data = closed_trades.to_dict(orient='records')
    return table_data


""" The actual tabs """


open_trade_tab = html.Div(
            [
                # CONTAINER FOR ENTERING A NEW TRADE
                html.Div(
                    [
                        html.H6('Add new trade'),
                        html.Div(
                            [
                                html.P('Pair:'),
                                dcc.Dropdown(id='pair', options=pairs, value='ETHUSDT'),
                            ],
                            style={'width': '20%', 'display': 'inline-block'}
                        ),
                        html.Div(
                            [
                                html.P('Entry:'),
                                dcc.Input(id='entry', placeholder=0.0, type='number', value=np.nan, min=0,
                                          style={'width':
                                                     '50px'})
                            ],
                            style={'width': '10%', 'display': 'inline-block'}
                        ),
                        html.Div(
                            [
                                html.P('Amount:'),
                                dcc.Input(id='size', placeholder=0.0, type='number', value=np.nan, min=0)
                            ],
                            style={'width': '10%', 'display': 'inline-block'}
                        ),
                        html.Div(
                            [
                                html.P('Stop loss:'),
                                dcc.Input(id='stop', placeholder=0.0, type='number', value=np.nan, min=0)
                            ],
                            style={'width': '10%', 'display': 'inline-block'}
                        ),
                        html.Div(
                            [
                                html.P('Trade type:'),
                                dcc.Dropdown(id='type', options=types, value='')
                            ],
                            style={'width': '20%', 'display': 'inline-block'}
                        ),
                        html.Div(
                            [
                                html.P('Direction:'),
                                dcc.RadioItems(id='direction', options=directions, value='')
                            ],
                            style={'width': '30%', 'display': 'inline-block'}
                        ),
                        html.Button('Open Trade', id='button')
                    ],
                    className="pretty_container twelve columns",
                    id="enter-trade"
                ),
                # CONTAINER FOR OPEN POSITIONS
                html.Div(
                    [
                        html.H6('Open Positions:'),
                        dash_table.DataTable(
                            id='open_table',
                            columns=open_trade_dict,
                            data=get_open_trades(diary),
                            style_table={
                                'height': '100px',
                                'overflow-y': 'scroll'
                            },
                            # You can use style conditional to color profitable and losing trades!
                            style_cell_conditional=[
                                {
                                    'if': {'column_id': c},
                                    'text-align': 'center'
                                } for c in ['pair', 'direction']
                            ],
                            style_as_list_view=True,
                            style_cell={'padding': '5px'},
                            style_header={'background-color': 'white', 'font-weight': 'bold'}
                        )
                    ],
                    className="pretty_container seven columns",
                    style={'margin-left': 0},
                    id="open-positions"
                ),
                # CONTAINER FOR CLOSING AN OPEN POSITION
                html.Div(
                    [
                        html.H6('CLOSE AN OPEN POSITION, OR MAYBE OPEN RISK AND OPEN P/L?')
                    ],
                    className="pretty_container four columns",
                    # style={},
                    id="close-position"
                )
            ]
        )

close_trade_tab = html.Div(
            [
                html.H6('Closed Trades:'),
                dash_table.DataTable(
                    id='closed_table',
                    columns=closed_trade_dict,
                    data=get_closed_trades(diary),
                    style_table={
                      'height': '250px',
                      'overflow-y': 'scroll'
                    },
                    # You can use style conditional to color profitable and losing trades!
                    style_cell_conditional=[
                        {
                            'if': {'column_id': c},
                            'text-align': 'center'
                        } for c in ['pair', 'direction']
                    ],
                    style_as_list_view=True,
                    style_cell={'padding': '5px'},
                    style_header={'background-color': 'white', 'font-weight': 'bold'}
                )
            ],
            className="pretty_container twelve columns"
        )
