# DASH application (GUI) for record keeping and inspecting performance.
# TODO:
#   1. Make a GUI to enter trade diary information, edit it and save it in a (csv?) file
#   2. Add capital tracking tab

 # TODO:
#   1. Set standard size for input boxes narrower
#   2. Fix horizontal alignment of Pair box.


import pandas as pd
import numpy as np
import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

""" Set up some stuff before defining the GUI """

diary = 'trading_diary.xlsx'
# writer = pd.ExcelWriter(path=record_file_path, datetime_format='DD-MM-YYY HH:MM', mode='a')

colors = {
    'background': '#313e5c',
    'text': '#fafafa'
}

pairs = [
    {'label': 'ETH/USDT', 'value': 'ETHUSDT'},
    {'label': 'BTC/USDT', 'value': 'BTCUSDT'},
    {'label': 'XRP/USDT', 'value': 'XRPUSDT'},
         ]


open_trade_cols = ['pair', 'size', 'entry', 'stop', 'direction']
open_trade_dict = [{'name': c, 'id': c} for c in open_trade_cols]


closed_trade_cols = ['pair', 'size', 'entry', 'exit', 'stop', 'P/L (USDT)', 'P/L (%)', 'pre capital',
                     'risk', 'rrr', 'direction', 'type', 'confidence', 'note']
# pre-cap not required here, but only for graphs
# maybe remove some other columns too.
closed_trade_dict = [{'name': c, 'id': c} for c in closed_trade_cols]


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


""" Write the app """
app = dash.Dash(__name__)

app.layout = html.Div(
    [
        html.Div(
            [
                html.H1("- TRADING RECORDS -",
                        style={'margin-bottom': '0px', 'color': colors['text'], 'text-align': 'center'}),
                html.H5("Good records are key to consistent profits",
                        style={'margin-top': '0px', 'color': colors['text'], 'text-align': 'center'})
            ]
        ),
        html.Div(
            [
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
                ),
            ],
        ),
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
                        dcc.Input(id='entry', placeholder=0.0, type='number', value=np.nan, min=0, style={'width':
                                                                                                              '50px'})
                    ],
                    style={'width': '20%', 'display': 'inline-block'}
                ),
                html.Div(
                    [
                        html.P('Amount:'),
                        dcc.Input(id='size', placeholder=0.0, type='number', value=np.nan, min=0)
                    ],
                    style={'width': '20%', 'display': 'inline-block'}
                )

            ],
            className="pretty_container twelve columns",
            id="enter-trade"
        ),
        # RECORD OF OLD TRADES:
        html.Div(
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
    ]
)


# app.layout = html.Div([
#     html.Label('Type of Trade'),
#     dcc.Dropdown(
#         options=[
#             {'label': 'New York City', 'value': 'NYC'},
#             {'label': u'Montréal', 'value': 'MTL'},
#             {'label': 'San Francisco', 'value': 'SF'}
#         ],
#         value='MTL'
#     ),
#     dcc.Input(id='my-id', value='initial value', type='text'),
#     html.Div(id='my-div')
#
# ])
#
#
# @app.callback(
#     Output(component_id='my-div', component_property='children'),
#     [Input(component_id='my-id', component_property='value')]
# )
# def update_output_div(input_value):
#     return 'You entered {}'.format(input_value)


if __name__ == '__main__':
    app.run_server(debug=True)
