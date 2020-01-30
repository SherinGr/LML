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
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

""" Set up some stuff before defining the GUI """

record_file_path = './trading_diary.xlsx'
writer = pd.ExcelWriter(path=record_file_path, datetime_format='DD-MM-YYY HH:MM', mode='a')

colors = {
    'background': '#313e5c',
    'text': '#fafafa'
}

pairs = [{'label': 'ETH/USDT', 'value': 'ETHUSDT'},
         {'label': 'BTC/USDT', 'value': 'BTCUSDT'},
         {'label': 'XRP/USDT', 'value': 'XRPUSDT'},
         ]


""" Write the app """
app = dash.Dash(__name__)

app.layout = html.Div(
    [
        html.Div(
            [
                html.H1("TRADING RECORDS",
                        style={'margin-bottom': '0px', 'color': colors['text'], 'text-align': 'center'}),
                html.H5("Good records are key to consistent profits",
                        style={'margin-top': '0px', 'color': colors['text'], 'text-align': 'center'})
            ]
        ),
        # CONTAINER FOR OPEN POSITIONS
        html.Div(
            [
                html.H2('OPEN POSITIONS')
            ],
            className="pretty_container five columns",
            style={'margin-left': 0},
            id="open-positions"
        ),
        # CONTAINER FOR CLOSING AN OPEN POSITION
        html.Div(
            [
                html.H2('CLOSE AN OPEN POSITION')
            ],
            className="pretty_container five columns",
            style={'width': '48%'},
            id="close-position"
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
            style={'width': '100%'},
            id="enter-trade"
        ),

    ],
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
