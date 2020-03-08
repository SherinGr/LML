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

import risk

current_capital = 1000  # HARDCODED FOR NOW:


""" Dictionaries for dropdowns etc """

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

open_trade_cols = ['pair', 'size', 'buy', 'stop', 'direction']
open_trade_dict = [{'name': c, 'id': c} for c in open_trade_cols]


""" Functions """


def open_trades(record_file, dict_output=False):
    trades = pd.read_excel(record_file, sheet_name='open')
    trades = trades.drop(columns=['date'])
    if dict_output:
        # For using this function in a dash table we need a dict as output:
        trades = trades.to_dict(orient='records')
    return trades


def open_risk_string():
    open_risk = sum(risk.trade_risk(current_capital, open_trades(diary)))

    if open_risk > risk.max_open_risk:
        color = 'red'
    else:
        color = 'green'

    return [html.Pre('Open risk is: \t'),
            html.P(' {:.2f}%'.format(open_risk), style={'color': color})]


def open_profit_string():
    profit = risk.open_profit(open_trades(diary))
    if profit < 0:
        color = 'red'
    else:
        color = 'green'
    return [html.Pre('Open profit/loss is: \t '),
            html.P('{:.2f}$'.format(profit), style={'color': color})]


""" The actual tab """


layout = html.Div(
            [
                # CONTAINER FOR ENTERING A NEW TRADE
                html.Div(
                    [
                        html.H6('Add New Trade:'),
                        html.Div(
                            [
                                html.P('Pair:'),
                                dcc.Dropdown(id='pair', options=pairs, value='ETHUSDT',
                                             style={'width': '97%', 'display': 'block', 'height': '40px'}),
                            ],
                            style={'width': '25%', 'display': 'inline-block'}
                        ),
                        html.Div(
                            [
                                html.P('Buy:'),
                                dcc.Input(id='buy', placeholder=0.0, type='number', value=np.nan, min=0,
                                          style={'width': '90%', 'display': 'inline', 'height': '40px'})
                            ],
                            style={'width': '15%', 'display': 'inline-block', 'vertical-align': 'top'}
                        ),
                        html.Div(
                            [
                                html.P('Amount:'),
                                dcc.Input(id='size', placeholder=0.0, type='number', value=np.nan, min=0,
                                          style={'width': '90%', 'height': '40px'})
                            ],
                            style={'width': '15%', 'display': 'inline-block', 'vertical-align': 'top'}
                        ),
                        html.Div(
                            [
                                html.P('Stop loss:'),
                                dcc.Input(id='stop', placeholder=0.0, type='number', value=np.nan, min=0,
                                          style={'width': '90%', 'height': '40px'})
                            ],
                            style={'width': '15%', 'display': 'inline-block', 'vertical-align': 'top'}
                        ),
                        html.Div(
                            [
                                html.P('Trade type:'),
                                dcc.Dropdown(id='type', options=types, value='',
                                             style={'width': '100%', 'display': 'block', 'height': '40px'})
                            ],
                            style={'width': '30%', 'display': 'inline-block'},
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.P('Confidence:'),
                                        dcc.Slider(id='confidence', min=1, max=3, step=None, value=2,
                                                   marks={1: 'Unsure', 2: 'OK', 3: 'Gonna Win!'})
                                    ],
                                    style={'width': '40%', 'display': 'block-inline', 'vertical-align': 'top'}
                                ),
                                html.Div(
                                    [
                                        html.P('Direction:'),
                                        dcc.RadioItems(id='direction', options=directions, value='',
                                                       style={'display': 'flex'})
                                    ],
                                    style={'display': 'block-inline'}
                                ),
                                html.Div(
                                    [
                                        html.Button('Enter Trade', id='button')
                                    ],
                                    style={'display': 'block', 'vertical-align': 'bottom'}
                                )
                            ],
                            style={'justify-content': 'space-between', 'display': 'flex'}
                        )
                    ],
                    className="pretty_container twelve columns",
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
                                data=open_trades(diary, dict_output=True),
                                style_table={
                                    'height': '126px',
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
                            ),
                            html.Div(
                                [
                                    html.Div(children=open_risk_string(),
                                             style={'display': 'flex'}),
                                    html.Div(children=open_profit_string(),
                                             style={'display': 'flex'})
                                ],
                                style={'justify-content': 'space-between', 'display': 'flex'}
                            )
                        ],
                        className="pretty_container seven columns",
                        style={'display': 'inline-block', 'margin-left': '0', 'margin-top': '0'},
                        id="open-positions"
                    ),
                    # CONTAINER FOR CALCULATING TRADE SIZE
                    html.Div(
                        [
                            html.H6('Calculate Position Size:'),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.P('Buy:'),
                                            dcc.Input(id='buy2', placeholder=0.0, type='number', value=np.nan, min=0,
                                                      style={'display': 'inline', 'width': '90%'})
                                        ],
                                        #  style={'display': 'inline-block'}
                                    ),
                                    html.Div(
                                        [
                                            html.P('Stop loss:'),
                                            dcc.Input(id='stop2', placeholder=0.0, type='number', value=np.nan, min=0,
                                                      style={'display': 'inline', 'width': '90%', })
                                        ],
                                        style={'display': 'inline-block'}
                                    ),
                                    html.Div(
                                        [
                                            html.P('Allowed size:', style={'display': 'block'}),
                                            dcc.Input(id='size2', placeholder=0.0, type='number', value=np.nan, min=0,
                                                      style={'display': 'inline', 'width': '90%'})
                                        ],
                                        style={'display': 'inline-block'}
                                    ),
                                ],
                                style={'justify-content': 'space-between', 'display': 'flex'}
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.P('Risk:'),
                                            dcc.RadioItems(id='risk', options=[{'label': '1%', 'value': 0.01},
                                                                               {'label': '2%', 'value': 0.02}],
                                                           value=0.01,
                                                           style={'display': 'flex'})
                                        ],
                                        style={'display': 'flex'}
                                    ),
                                    html.Div(
                                        [
                                            html.P('Use Leverage:'),
                                            dcc.RadioItems(id='leverage', options=[{'label': 'No', 'value': 1},
                                                                                   {'label': 'Yes', 'value': 5}],
                                                           value=0.01,
                                                           style={'display': 'flex'})
                                        ],
                                        style={'display': 'flex'}
                                    ),

                                ], style={'display': 'inline'}
                            )
                            # TODO:
                            #   1. Layout
                            #   4. Button to calculate

                        ],
                        className="pretty_container four columns",
                        style={'display': 'inline', 'margin-right': '0', 'margin-left': '0', 'margin-top': '0'},
                        id="calculate_size"
                    )
                ])
            ]
        )


@app.callback([Output('open_table', 'data'),
               Output('buy', 'value'),
               Output('size', 'value'),
               Output('stop', 'value'),
               Output('type', 'value'),
               Output('direction', 'value'),
               Output('confidence', 'value')],
              [Input('button', 'n_clicks')],
              [State('pair', 'value'),
               State('buy', 'value'),
               State('size', 'value'),
               State('stop', 'value'),
               State('type', 'value'),
               State('direction', 'value'),
               State('confidence', 'value')]
              )
def submit_trade(clicks, pair, buy, size, stop, idea, direction, confidence):
    if clicks is None:
        pass
    elif any(x == 0 for x in [buy, size, stop]) or idea == '' or direction == '':
        # TODO: trade incomplete, MAKE COLOR OF MISSING BOX RED
        return '#ff0000'
    else:
        # Add new trade at the top of the diary excel file:
        index = pd.DatetimeIndex([datetime.datetime.now()])
        # trade = pd.DataFrame(columns=open_trade_dict, index=index)
        trade = pd.DataFrame({
            'pair': pair, 'buy': buy, 'size': size, 'stop': stop, 'type': idea, 'direction': direction,
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

        # TODO: Animate button for visual confirmation
        trades = open_trades(diary)
        return trades, 0, 0, 0, '', '', 2
