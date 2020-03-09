import pandas as pd
import dash_table
import dash_html_components as html
import dash_core_components as dcc

from app import app
from app import diary

from tabs.open import *

closed_trade_cols = ['pair', 'size', 'entry', 'exit', 'stop', 'P/L (%)',
                     'risk (%)', 'RRR', 'direction', 'type', 'confidence', 'note']
closed_trade_dict = [{'name': c, 'id': c} for c in closed_trade_cols]


def get_closed_trades(record_file):
    closed_trades = pd.read_excel(record_file, sheet_name='closed')
    # closed_trades = closed_trades.drop(columns=['date'])
    # TODO: make sure to drop the right columns here to match the table dict!
    table_data = closed_trades.to_dict(orient='records')
    return table_data


# TODO: Update capital on each trade that is closed, save in df.
# TODO: Save latest capital for each day in a separate df


layout = html.Div(
    [
        html.Div(
            [
                html.Div(
                    [
                        html.H6('Open Positions:', style={'margin-bottom': '10px'}),
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
                            row_selectable='single',
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
                        html.H6('Close Position:'),
                        html.P('Exit:'),
                        dcc.Input(id='exit'),
                        html.P('Note:'),
                        dcc.Input(id='note')
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
            className='pretty_container twelve columns',
            style={'margin-top': '0'}
        )

    ],
)
