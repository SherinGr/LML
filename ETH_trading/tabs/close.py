import pandas as pd
import dash_table
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State

from app import app, user_data
import tradelib as tl

layout = html.Div(
    [
        html.Div(
            [
                html.Div(
                    [
                        html.H6('Select A Position To Close:', style={'margin-bottom': '10px'}),
                        dash_table.DataTable(
                            id='open_table2',
                            columns=tl.open_trade_dict,
                            data=tl.read_trades(user_data['diary_file'], 'open', dict_output=True),
                            style_table={
                                'height': '126px',
                                'overflow-y': 'scroll',
                            },
                            style_cell_conditional=[
                                {
                                    'if': {'column_id': c},
                                    'text-align': 'center'
                                } for c in ['pair', 'direction']

                            ],
                            row_selectable='single',
                            style_as_list_view=True,
                            style_cell={'padding': '5px'},
                            style_header={'background-color': 'white', 'font-weight': 'bold'}
                        )
                    ],
                    className='pretty_container six columns',
                    style={'margin-left': '0'}
                ),
                html.Div(
                    [
                        html.H6('Close Position:', style={'margin-bottom': '20px'}),
                        html.Div(
                            [
                                html.P('Exit price:'),
                                dcc.Input(id='exit', placeholder=0, type='number', min=0,
                                          style={'width': '20%'}),
                                html.Button('Close Trade', id='close_trade_button')
                            ],
                            style={'display': 'flex', 'justify-content': 'space-between',
                                   'vertical-align': 'center'}
                        ),
                        html.P('Note:', style={'margin-top': '10px'}),
                        dcc.Input(id='note', style={'width': '100%'})
                    ],
                    className='pretty_container six columns',
                    style={'margin-right': '0', 'margin-left': '0'}
                )
            ],
            className='row flex-display',
        ),
        html.Div(
            [
                html.H6('Closed Trades:'),
                dash_table.DataTable(
                    id='closed_table',
                    columns=tl.closed_trade_dict,
                    data=tl.read_trades(user_data['diary_file'], 'closed', dict_output=True),
                    style_table={
                      'height': '270px',
                      'overflow-y': 'scroll'
                    },
                    # You can use style conditional to color profitable and losing trades!
                    style_cell_conditional=[
                        {
                            'if': {'column_id': c},
                            'text-align': 'center'
                        } for c in ['pair', 'direction', 'type']
                    ],
                    style_data_conditional=[
                        {
                            'if': {
                                'column_id': 'P/L (%)',
                                'filter_query': '{P/L (%)} > 0',
                            },
                            'backgroundColor': '#3D9970',
                            'color': 'white',
                        },
                        {
                            'if': {
                                'column_id': 'P/L (%)',
                                'filter_query': '{P/L (%)} < 0',
                            },
                            'backgroundColor': '#A83232',
                            'color': 'white',
                        }
                    ],
                    style_as_list_view=True,
                    style_cell={'padding': '5px'},
                    style_header={'background-color': 'white', 'font-weight': 'bold'}
                )
            ],
            className='pretty_container twelve columns',
            style={'margin-top': '0'}
        )

    ],
)


# @app.callback(Output('selected_trade', 'children'),
#               [Input('open_table2', 'rows'),
#                Input('open_table2', 'selected_row_indices')])
# def open_trade_selected(rows, selected_row):
#     # TODO:
#     #   1. retrieve the selected trade
#     #   2. Show it in the close trade table
#
#     pass


# @app.callback(Output('closed_table', 'data'),
#               [Input('close_trade_button', 'n_clicks')],
#               [State('open_table2', 'data'),
#                State('open_table2', 'selected_rows'),
#                State('exit', 'value'),
#                State('note', 'value')])
# def close_trade(clicks, open_trades, selected_trade, exit, note):
#     if clicks is None:
#         pass
#     else:
#         pass
#     # TODO:
#     #   1. calculate values on closing a trade
#     #   1b. Update per_trade_capital dataframe!!!
#     #   1c. Make sure to add the timespan
#     #   2. add to diary,
#     #   2b. remove from open !!!
#     #   3. display in table
