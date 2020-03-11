import numpy as np
import pandas as pd
import datetime

from openpyxl import load_workbook

import dash_table
import dash_core_components as dcc
import dash_html_components as html

from dash.dependencies import Input, Output, State

import risk

from app import app, user_data, client

""" Dictionaries for dropdowns etc """

pairs = [
    {'label': 'ETH/USDT', 'value': 'ETHUSDT'},
    {'label': 'BTC/USDT', 'value': 'BTCUSDT'},
    {'label': 'XRP/USDT', 'value': 'XRPUSDT'},
         ]

types = [
    {'label': 'pullback to value', 'value': 'pullback to value'},
    {'label': 'ATR extreme', 'value': 'ATR extreme'},
    {'label': 'price rejection', 'value': 'price rejection'}
]

directions = [
    {'label': 'LONG', 'value': 'LONG'},
    {'label': 'SHORT', 'value': 'SHORT'}
]

open_trade_cols = ['pair', 'size', 'entry', 'stop', 'direction']
open_trade_dict = [{'name': c, 'id': c} for c in open_trade_cols]


""" Functions """


def open_trades(record_file, dict_output=False):
    trades = pd.read_excel(record_file, sheet_name='open')
    trades = trades.drop(columns=['date'])
    if dict_output:
        # For using this function in a dash table we need a dict as output:
        trades = trades.to_dict(orient='records')
    return trades


def write_open_trade_to_records(record_file, trade):
    with pd.ExcelWriter(path=record_file, engine='openpyxl', datetime_format='DD-MM-YYYY hh:mm', mode='a') as \
            writer:
        # Open the file:
        writer.book = load_workbook(record_file)
        # Copy existing sheets:
        writer.sheets = dict((ws.title, ws) for ws in writer.book.worksheets)
        # Add new trade on top of the existing data:
        sheet = 'open'
        writer.book[sheet].insert_rows(2)
        trade.to_excel(writer, sheet_name=sheet, startrow=1, header=None, index_label='date')
        writer.close()


def open_risk_string():
    current_capital = user_data['capital'][-1]
    open_risk = sum(risk.trade_risk(current_capital, open_trades(user_data['diary_file'])))

    if open_risk > risk.max_open_risk:
        color = 'red'
    else:
        color = 'green'

    return [html.Pre('Open risk: \t'),
            html.P(' {:.2f}%'.format(open_risk), style={'color': color})]


def open_profit_string():
    profit = risk.open_profit(open_trades(user_data['diary_file']))
    percent = profit/user_data['capital'][-1]*100
    if profit < 0:
        color = 'red'
    else:
        color = 'green'
    return [html.Pre('Open profit/loss: \t '),
            html.P('{:.2f}$ ({:.2f}%)'.format(profit, percent), style={'color': color})]


""" The actual tab """


layout = html.Div(
            [
                # CONTAINER FOR ENTERING A NEW TRADE
                dcc.Interval(id='price_call'),
                html.Div([
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
                                html.P('Amount:'),
                                dcc.Input(id='qty', placeholder=0.0, type='number', value=np.nan, min=0,
                                          style={'width': '90%', 'display': 'inline', 'height': '40px'})
                            ],
                            style={'width': '15%', 'display': 'inline-block', 'vertical-align': 'top'}
                        ),
                        html.Div(
                            [
                                html.P('Entry:'),
                                dcc.Input(id='entry', placeholder=0.0, type='number', value=np.nan, min=0,
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
                                        # It would be possible to determine SHORT/LONG from entry and stop!
                                        html.P('Direction:'),
                                        dcc.RadioItems(id='direction', options=directions, value='',
                                                       style={'display': 'flex', 'margin-top': '10px',
                                                              'margin-bottom': '0'},
                                                       labelStyle={'margin-right': '10px'})
                                    ],
                                    style={'display': 'block-inline'}
                                ),
                                html.Div(
                                    [
                                        html.Button('Enter Trade', id='enter_trade_button')
                                    ],
                                    style={'display': 'block', 'align-items': 'flex-end', 'margin-top': '35px'}
                                )
                            ],
                            style={'justify-content': 'space-between', 'display': 'flex'}
                        )
                    ],
                    className="pretty_container twelve columns",
                    style={'margin-right': '0'},
                    id="enter-trade"
                    ),
                    ],
                    className='row flex-display'
                ),
                # CONTAINER FOR OPEN POSITIONS
                html.Div([
                    html.Div(
                        [
                            html.H6('Open Positions:', style={'margin-bottom': '10px'}),
                            dash_table.DataTable(
                                id='open_table',
                                columns=open_trade_dict,
                                data=open_trades(user_data['diary_file'], dict_output=True),
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
                                    html.Div(id='open_risk', children=open_risk_string(),
                                             style={'display': 'flex'}),
                                    html.Div(id='open_profit', children=open_profit_string(),
                                             style={'display': 'flex', 'width': '60%'})
                                ],
                                style={'justify-content': 'space-between', 'display': 'flex',
                                       'margin-top': '10px'}
                            )
                        ],
                        className="pretty_container seven columns",
                        style={'margin-left': '0', 'margin-top': '0'},
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
                                            html.P('Entry:'),
                                            dcc.Input(id='entry2', placeholder=0.0, type='number', value=np.nan,
                                                      min=0,
                                                      style={'display': 'inline', 'width': '90%'})
                                        ],
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
                                            dcc.Input(id='qty2', placeholder=0.0, type='number', value=np.nan,
                                                      min=0,
                                                      readOnly=True,
                                                      style={'display': 'inline', 'width': '100%'})
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
                                            html.Pre('Risk:'),
                                            dcc.RadioItems(id='risk', options=[{'label': '1%  ', 'value': 0.01},
                                                                               {'label': '2%', 'value': 0.02}],
                                                           value=0.01,
                                                           style={'display': 'flex', 'margin-right': '2px'},
                                                           labelStyle={'margin-right': '24px'})
                                        ],
                                        style={'display': 'flex', 'justify-content': 'space-between',
                                               'margin-top': '10px'}
                                    ),
                                    html.Div(
                                        [
                                            html.P('Use Leverage:'),
                                            dcc.RadioItems(id='leverage', options=[{'label': 'Yes', 'value': 5},
                                                                                   {'label': 'No', 'value': 1}],
                                                           value=1,
                                                           style={'display': 'flex', 'margin-right': '10px'},
                                                           labelStyle={'margin-right': '20px'})
                                        ],
                                        style={'display': 'flex', 'justify-content': 'space-between'}
                                    ),
                                    html.Div(
                                        [
                                            html.Button('GO!', id='calc_size_button')
                                        ],
                                        style={'display': 'flex', 'justify-content': 'flex-end'}
                                    )
                                ], style={'display': 'inline'}
                            )
                        ],
                        className="pretty_container five columns",
                        style={'margin-right': '0', 'margin-left': '0', 'margin-top': '0'},
                        id="calculate_size"
                    )
                    ],
                    className='row flex-display'
                )
            ]
        )

# Callback for continuous update of open profit:
@app.callback(Output('open_profit', 'children'), [Input('price_call', 'n_intervals')])
def update_profit(_):
    return open_profit_string()


# Callback for calculate trade size button:
@app.callback(Output('qty2', 'value'),
              [Input('calc_size_button', 'n_clicks')],
              [State('entry2', 'value'),
               State('stop2', 'value'),
               State('risk', 'value'),
               State('leverage', 'value')])
def calculate_size(clicks, entry, stop, max_risk, leverage):
    if clicks is None:
        pass
    else:
        cap = user_data['capital'][-1]
        return round(risk.max_qty(cap, entry, stop, max_risk, leverage), 4)


# Callback for enter trade button:
# TODO: Handle error when input is not complete yet
@app.callback([Output('open_table', 'data'),
               Output('entry', 'value'),
               Output('qty', 'value'),
               Output('stop', 'value'),
               Output('type', 'value'),
               Output('direction', 'value'),
               Output('confidence', 'value'),
               Output('open_risk', 'children')],
              [Input('enter_trade_button', 'n_clicks')],
              [State('pair', 'value'),
               State('entry', 'value'),
               State('qty', 'value'),
               State('stop', 'value'),
               State('type', 'value'),
               State('direction', 'value'),
               State('confidence', 'value')]
              )
def submit_trade(clicks, pair, entry, qty, stop, idea, direction, confidence):
    if clicks is None:
        pass
    elif any(x == 0 for x in [entry, qty, stop]) or idea == '' or direction == '':
        return 0
    else:
        # Add new trade at the top of the diary excel file:
        index = pd.DatetimeIndex([datetime.datetime.now()])
        trade = pd.DataFrame({
            'pair': pair, 'size': qty, 'entry': entry, 'stop': stop, 'type': idea, 'direction': direction,
            'confidence':
                confidence
        }, index=index)

        write_open_trade_to_records(user_data['diary_file'], trade)
        trades = open_trades(user_data['diary_file'], dict_output=True)

        return trades, 0, 0, 0, '', '', 2, open_risk_string()
