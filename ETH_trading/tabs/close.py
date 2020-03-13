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
                html.Div(id='selected_trade', style={'visibility': 'hidden'}),
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
                            selected_rows=[0],
                            style_as_list_view=True,
                            style_cell={'padding': '5px'},
                            style_header={'background-color': 'white', 'font-weight': 'bold'}
                        )
                    ],
                    className='pretty_container seven columns',
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
                    className='pretty_container five columns',
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


@app.callback(Output('closed_table', 'data'),
              [Input('close_trade_button', 'n_clicks')],
              [State('open_table2', 'selected_rows'),
               State('exit', 'value'),
               State('note', 'value')])
def close_trade(clicks, selected_row, close, note):
    if clicks is None or not selected_row:
        pass
    else:
        record_file = user_data['diary_file']
        open_trades = tl.read_trades(record_file, 'open')
        trade = open_trades.iloc[selected_row]
        # Compute features of the closed trade:
        closed_trade = tl.fill_trade(trade, close, note)
        # Write the trade to the records:
        tl.write_trade_to_records(record_file, 'closed', closed_trade)
        # Remove the open trade from the open trade records:
        # TODO
        # Update the running average features:
        tl.update_user_data(closed_trade)

        closed_trades = tl.read_trades(record_file, 'closed', dict_output=True)
        return closed_trades
