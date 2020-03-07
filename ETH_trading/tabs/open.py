import numpy as np
import pandas as pd
import datetime

from openpyxl import load_workbook

import dash_table
import dash_core_components as dcc
import dash_html_components as html

from dash.dependencies import Input, Output, State

from app import app
from app import diary


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


""" Functions to read trades from the records """


def get_open_trades(record_file):
    open_trades = pd.read_excel(record_file, sheet_name='open')
    open_trades = open_trades.drop(columns=['date'])
    table_data = open_trades.to_dict(orient='records')
    return table_data


""" The actual tab """


layout = html.Div(
            [
                # CONTAINER FOR ENTERING A NEW TRADE
                html.Div(
                    [
                        html.H6('Add new trade:'),
                        html.Div(
                            [
                                html.P('Pair:'),
                                dcc.Dropdown(id='pair', options=pairs, value='ETHUSDT',
                                             style={'width': '97%', 'display': 'block'}),
                            ],
                            style={'width': '25%', 'display': 'inline-block'}
                        ),
                        html.Div(
                            [
                                html.P('Entry:'),
                                dcc.Input(id='entry', placeholder=0.0, type='number', value=np.nan, min=0,
                                          style={'width': '90%'})
                            ],
                            style={'width': '15%', 'display': 'inline-block'}
                        ),
                        html.Div(
                            [
                                html.P('Amount:'),
                                dcc.Input(id='size', placeholder=0.0, type='number', value=np.nan, min=0,
                                          style={'width': '90%'})
                            ],
                            style={'width': '15%', 'display': 'inline-block'}
                        ),
                        html.Div(
                            [
                                html.P('Stop loss:'),
                                dcc.Input(id='stop', placeholder=0.0, type='number', value=np.nan, min=0,
                                          style={'width': '90%'})
                            ],
                            style={'width': '15%', 'display': 'inline-block'}
                        ),
                        html.Div(
                            [
                                html.P('Trade type:'),
                                dcc.Dropdown(id='type', options=types, value='',
                                             style={'width': '100%', 'display': 'block'})
                            ],
                            style={'width': '30%', 'display': 'inline-block'},
                        ),

                        html.Div(
                            [
                                html.P('Confidence:'),
                                dcc.Slider(id='confidence', min=1, max=3, step=None, value=2,
                                           marks={1: 'Unsure', 2: 'OK', 3: 'Gonna Win!'})
                            ],
                            style={'width': '40%', 'display': 'inline-block'}
                        ),
                        html.Div(
                            [
                                html.P('Direction:'),
                                dcc.RadioItems(id='direction', options=directions, value='',
                                               style={'display': 'flex'})
                            ],
                            style={'width': '25%', 'display': 'inline-block'}
                        ),
                        html.Div(
                            [
                                html.Button('Enter Trade', id='button')
                            ],
                            style={'width': '15%', 'display': 'inline-block'}
                        )
                    ],
                    className="pretty_container twelve columns",
                    #style={'display': 'flex'},
                    id="enter-trade"
                ),
                # CONTAINER FOR OPEN POSITIONS
                html.Div([
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
                        style={'display': 'inline-block', 'margin-left': '0', 'margin-top': '0'},
                        id="open-positions"
                    ),
                    # CONTAINER FOR CLOSING AN OPEN POSITION
                    html.Div(
                        [
                            html.H6('CLOSE AN OPEN POSITION, OR MAYBE OPEN RISK AND OPEN P/L?')

                        ],
                        className="pretty_container",
                        style={'display': 'flex', 'margin-right': '0'},
                        id="close-position"
                    )
                ])
            ]
        )


@app.callback([Output('open_table', 'data'),
               Output('entry', 'value'),
               Output('size', 'value'),
               Output('stop', 'value'),
               Output('type', 'value'),
               Output('direction', 'value'),
               Output('confidence', 'value')],
              [Input('button', 'n_clicks')],
              [State('pair', 'value'),
               State('entry', 'value'),
               State('size', 'value'),
               State('stop', 'value'),
               State('type', 'value'),
               State('direction', 'value'),
               State('confidence', 'value')]
              )
def submit_trade(clicks, pair, entry, size, stop, idea, direction, confidence):
    if clicks is None:
        pass
    elif any(x == 0 for x in [entry, size, stop]) or idea == '' or direction == '':
        # TODO: trade incomplete, MAKE COLOR OF BUTTON RED
        return '#ff0000'

    else:
        # TODO: Change . into , for excell

        # Add new trade at the top of the diary excel file:
        index = pd.DatetimeIndex([datetime.datetime.now()])
        # trade = pd.DataFrame(columns=open_trade_dict, index=index)
        trade = pd.DataFrame({
            'pair': pair, 'entry': entry, 'size': size, 'stop': stop, 'type': idea, 'direction': direction,
            'confidence':
                confidence
        }, index=index)

        with pd.ExcelWriter(path=diary, engine='openpyxl', datetime_format='DD-MM-YYYY hh:mm', mode='a') as \
                writer:
            # Open the file:
            writer.book = load_workbook(diary)
            # Copy existing sheets:
            writer.sheets = dict((ws.title, ws) for ws in writer.book.worksheets)
            # Add new trade on top of the existing data:
            sheet = 'open'
            writer.book[sheet].insert_rows(2)
            trade.to_excel(writer, sheet_name=sheet, startrow=1, header=None, index_label='date')
            writer.close()

        # TODO: Reset input values:
        # TODO: Animate button for visual confirmation
        # TODO: Update table data
        open_trades = get_open_trades(diary)
        return open_trades, 0, 0, 0, '', '', 2
