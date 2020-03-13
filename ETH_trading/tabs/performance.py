import dash_core_components as dcc
import dash_html_components as html

from app import app, user_data
import tradelib as tl

cap = user_data['capital'][-1]
no_trades = len(user_data['capital'])
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


def target_capital_data(n_days):
    d = tl.capital_target(n_days)
    data = dict(
        x=d.index,
        y=d.values,
        name='Target',
        line=dict(shape='spline', width=2, color='#8dc270')
    )
    return data


layout = html.Div(
    [
        html.Div(
            [
                html.Div(
                    [
                        html.H5('Trades and Profits', style={'line-height': '0.5'}),
                        dcc.RadioItems(id='timespan', options=tl.spans, value='D',
                                       style={'display': 'flex', 'margin-top': '0px',
                                              'margin-bottom': '0'},
                                       labelStyle={'margin-right': '10px'}),
                        html.P('Table with daily, weekly or monthly trades and total profit')
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
                                                dcc.Input(id='monthly_target_profit', type='number', value='10',
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
              html.H1('MONTHLY/DAILY/WEEKLY STATS')
            ],
            className='pretty_container flex-display',
            style={'margin-left': '0', 'margin-right': '0', 'margin-top': '0'}
        ),
        html.Div(
            [
                html.Div(
                    [
                      html.H1('SHORTLONG STACKED BARPLOT')
                    ],
                    className='pretty_container six columns',
                    style={'margin-left': '0', 'margin-top': '0', 'margin-right': '0'}
                ),
                html.Div(
                    [
                      html.H1('TIMESPAN STACKED DISTPLOT')
                    ],
                    className='pretty_container six columns',
                    style={'margin-top': '0', 'margin-right': '0'}
                )
            ],
            className='row flex-display'
        )
    ]
)
