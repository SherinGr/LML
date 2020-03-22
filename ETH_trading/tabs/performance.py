import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output, State

import plotly.figure_factory as ff

from app import app, user_data
import tradelib as tl

cap = user_data['capital'][-1]
no_trades = len(user_data['capital']) - 1
avg_profit = user_data['avg_profit'][-1]
wl_rate = user_data['win_rate'][-1]
expect = user_data['expectancy'][-1]

y_scale_options = [
    {'label': 'lin', 'value': 'lin'},
    {'label': 'log', 'value': 'log'}
]

x_scale_options = [
    {'label': 'trade', 'value': 'T'},
    {'label': 'day', 'value': 'D'}
]

stats_table_cols = [{'name': 'period', 'id': 'period'},
                    {'name': 'trades', 'id': 'trades'},
                    {'name': 'profit', 'id': 'P/L'}]

def target_capital_data(n_days):
    d = tl.capital_target(n_days)
    data = dict(
        x=d.index,
        y=d.values,
        name='Target',
        line=dict(shape='spline', width=2, color='#8dc270')
    )
    return data


def shortlong_data():
    trades = tl.read_trades(user_data['diary_file'], 'closed')
    shorts = trades[trades['direction'] == 'SHORT']
    longs = trades[trades['direction'] == 'LONG']

    win_s = sum(shorts['P/L (%)'] >= 0)
    win_l = sum(longs['P/L (%)'] >= 0)

    lose_s = sum(shorts['P/L (%)'] < 0)
    lose_l = sum(longs['P/L (%)'] < 0)

    graph_data = [
        {'x': ['SHORT', 'LONG'],
         'y': [win_s, win_l],
         'name': 'won',
         'type': 'bar',
         'marker': {'color': '#3D9970'}
         },
        {'x': ['SHORT', 'LONG'],
         'y': [lose_s, lose_l],
         'name': 'lost',
         'type': 'bar',
         'marker': {'color': '#A83232'}
         }
    ]
    return graph_data


def timespan_data():
    pass


layout = html.Div(
    [
        html.Div(
            [
                html.Div(
                    [
                        html.H5('Trades and Profits', style={'line-height': '0.5'}),
                        dcc.RadioItems(id='timespan', options=tl.spans, value='D',
                                       style={'display': 'flex', 'margin-top': '0px',
                                              'margin-bottom': '10px'},
                                       labelStyle={'margin-right': '10px'}),
                        dash_table.DataTable(
                                id='stats_table',
                                columns=stats_table_cols,
                                data=[],
                                # style_table={
                                #     'height': '400px',
                                #     'overflow-y': 'scroll'
                                # },
                                # You can use style conditional to color profitable and losing trades!
                                style_cell_conditional=[{
                                        'if': {'column_id': 'period'},
                                        'text-align': 'left'
                                }],
                                style_as_list_view=True,
                                persistence=True,
                                persisted_props=['data'],
                                style_cell={'padding': '5px'},
                                style_header={'background-color': 'white', 'font-weight': 'bold'}
                            ),
                    ],
                    className='pretty_container four columns',
                    style={'margin-left': '0'}
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.H5(str(no_trades), style={'margin': '0'}),
                                        html.P('No. Trades', style={'margin': '0'})
                                    ],
                                    className='mini_container',
                                    style={'margin-left': '0', 'flex-grow': '1'}
                                ),
                                html.Div(
                                    [
                                        html.H5('{:.2f}%'.format(avg_profit), style={'margin': '0'}),
                                        html.P('Avg. Profit', style={'margin': '0'})
                                    ],
                                    className='mini_container',
                                    style={'margin-left': '0', 'flex-grow': '1'}
                                ),
                                html.Div(
                                    [
                                        html.H5('{:.2f}%'.format(wl_rate*100), style={'margin': '0'}),
                                        html.P('Win Rate', style={'margin': '0'})
                                    ],
                                    className='mini_container',
                                    style={'margin-left': '0', 'flex-grow': '1'}
                                ),
                                html.Div(
                                    [
                                        html.H5('{:.2f}'.format(expect), style={'margin': '0'}),
                                        html.P('Expectancy', style={'margin': '0'})
                                    ],
                                    className='mini_container',
                                    style={'margin-left': '0', 'flex-grow': '1'}
                                ),
                                html.Div(
                                    [
                                        html.H5('{:.2f}'.format(cap), style={'margin': '0'}),
                                        html.P('Capital ($)', style={'margin': '0'})
                                    ],
                                    className='mini_container',
                                    style={'margin-left': '0', 'margin-right': '0', 'flex-grow': '1'}
                                )
                            ],
                            className='row container-display',
                            style={'display': 'flex-inline', 'justify-content': 'space-between', 'margin-right': '0'}
                        ),
                        html.Div(
                            [
                                html.H5('Capital Tracking', style={'text-align': 'center', 'line-height': '0.5'}),
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                dcc.RadioItems(id='y_opt', options=y_scale_options, value='lin',
                                                               style={'display': 'flex', 'margin-top': '0px',
                                                                      'margin-bottom': '0'},
                                                               labelStyle={'margin-right': '10px'}),
                                            ],
                                        ),
                                        html.Div(
                                            [
                                                html.P('Monhtly profit target (%):', style={'display': 'inline',
                                                                                            'margin-right': '10px'}),
                                                dcc.Input(id='monthly_target_profit', type='number', value=10,
                                                          style={'margin': '0', 'width': '50px', 'display':
                                                                 'inline', 'height': '25px'})
                                                # dcc.RadioItems(id='x_opt', options=x_scale_options, value='D',
                                                #                style={'display': 'flex', 'margin-top': '0px',
                                                #                       'margin-bottom': '0'},
                                                #                labelStyle={'margin-right': '10px'}),
                                            ],
                                            style={'margin-right': '15px'}
                                        )
                                    ],
                                    style={'display': 'flex', 'justify-content': 'space-between'}
                                ),
                                html.Div(
                                    [
                                        dcc.Graph(
                                            id='capital_graph',
                                            figure={
                                                # Data is a list of dicts, each dict with x,y-data.
                                                'data': [
                                                    {
                                                        'x': user_data['capital'].index,
                                                        'y': user_data['capital'].values,
                                                        'name': 'Capital',
                                                        'line': dict(shape='linear', width=2,
                                                                     color='#9DDCFA')
                                                    },
                                                    target_capital_data(20)
                                                ],
                                                'layout': dict(
                                                    # xaxis={'title': 'Date'},
                                                    yaxis={'type': 'lin', 'title': 'Capital ($)'},
                                                    margin={'l': 50, 'r': 15, 't': 10, 'b': 35},
                                                    height=350,
                                                    legend={'x': 0.04, 'y': 0.96},
                                                    paper_bgcolor='#e8e8e8',
                                                    font={'family': 'Dosis', 'size': 13}
                                                )
                                            }
                                        )
                                    ],
                                    style={'width': '100%'}
                                ),
                            ],
                            className='pretty_container',
                            style={'margin-left': '0', 'margin-right': '0', 'margin-top': '0'}
                        )
                    ],
                    className='eight columns',
                    style={'flex-grow': '1', 'margin-left': 0}
                ),
             ],
            className='row flex-display'
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.H5('Periodic Performance Indicators', style={'line-height': '0.5'})
                    ],
                    className='pretty_container eight columns',
                    style={'margin-left': '0', 'margin-top': '0', 'margin-right': '0'}
                ),
                html.Div(
                    [
                        html.H5('Short/Long Performance', style={'line-height': '0.5', 'margin-bottom': '10'}),
                        dcc.Graph(
                            id='shortlong_graph',
                            figure={
                                'data': shortlong_data(),
                                'layout': dict(
                                    barmode='stack',
                                    margin={'l': 20, 'r': 15, 't': 10, 'b': 25},
                                    height=250,
                                    paper_bgcolor='#e8e8e8',
                                    font={'family': 'Dosis', 'size': 13},
                                    showlegend=False,
                                )
                            }
                        )
                    ],
                    className='pretty_container four columns',
                    style={'margin-top': '0', 'margin-right': '0'}
                )
            ],
            className='row flex-display',
        ),
        html.Div(
            [

                html.Div(
                    [
                      html.H1('TIMESPAN STACKED DISTPLOT', style={'line-height': '0.5'})
                    ],
                    className='pretty_container six columns',
                    style={'margin-left': '0', 'margin-top': '0', 'margin-right': '0'}
                ),
                html.Div(
                    [
                      html.H1('FEATURE OVER TIME PLOT', style={'line-height': '0.5'})
                    ],
                    className='pretty_container six columns',
                    style={'margin-top': '0', 'margin-right': '0'}
                )
            ],
            className='row flex-display'
        )
    ]
)


@app.callback(Output('capital_graph', 'figure'),
              [Input('monthly_target_profit', 'value'),
               Input('y_opt', 'value')]
              )
def update_capital_target(target_profit, y_scale_type):
    figure = {
        # Data is a list of dicts, each dict with x,y-data.
        'data': [
            {
                'x': user_data['capital'].index,
                'y': user_data['capital'].values,
                'name': 'Capital',
                'line': dict(shape='linear', width=2,
                             color='#9DDCFA')
            },
            target_capital_data(target_profit)
        ],
        'layout': dict(
            # xaxis={'title': 'Date'},
            yaxis={'type': y_scale_type, 'title': 'Capital ($)'},
            margin={'l': 50, 'r': 15, 't': 10, 'b': 35},
            height=350,
            legend={'x': 0.04, 'y': 0.96},
            paper_bgcolor='#e8e8e8',
            font={'family': 'Dosis', 'size': 13}
        )
    }
    return figure
