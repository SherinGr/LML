import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output, State

from datetime import datetime
import pandas as pd

from app import app, user_data
import tradelib as tl

y_scale_options = [
    {'label': 'lin', 'value': 'lin'},
    {'label': 'log', 'value': 'log'}
]

x_scale_options = [
    {'label': 'trade', 'value': 'T'},
    {'label': 'day', 'value': 'D'}
]

summary_table_cols = [{'name': 'period', 'id': 'period'},
                      {'name': '#', 'id': 'trades'},
                      {'name': 'P/L (%)', 'id': 'P/L (%)'}]

features = [
    {'label': 'Average profit', 'value': 'avg_profit'},
    {'label': 'Expectancy', 'value': 'expectancy'},
    {'label': 'Risk reward ratio', 'value': 'avg_rrr'},
    {'label': 'Win rate', 'value': 'win_rate'},
    {'label': 'Timespan', 'value': 'avg_timespan'}
]


def target_capital_data(n_days, month_profit_target):
    d = tl.capital_target(n_days, month_profit_target)
    data = dict(
        x=d.index,
        y=d.values,
        name='Target',
        line=dict(shape='spline', width=2, color='#8dc270')
    )
    return data


def shortlong_figure():
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

    figure = {
        'data': graph_data,
        'layout': dict(
            barmode='stack',
            margin={'l': 20, 'r': 15, 't': 10, 'b': 15},
            height=250,
            paper_bgcolor='#e8e8e8',
            font={'family': 'Dosis', 'size': 13},
            showlegend=False,
        )
    }
    return figure


def timespan_figure():
    # Extract the timespans of winners and losers from the records:
    trades = tl.read_trades(user_data['diary_file'], 'closed')
    data = trades[['P/L (%)', 'timespan (min)']]

    winners = data[data['P/L (%)'] >= 0]
    losers = data[data['P/L (%)'] < 0]

    win_spans = winners.drop(columns=['P/L (%)'])/60  # convert to hours
    lose_spans = losers.drop(columns=['P/L (%)'])/60  # convert to hours

    # Plot the histogram of this data:
    graph_data = [
        {
            'x': list(win_spans.values.squeeze()),
            'name': 'won',
            'type': 'histogram',
            'nbins': 10,
            'histnorm': 'percent',
            'opacity': '0.75',
            'bingroup': 1,
            'marker': {'color': '#3D9970'}
        },
        {
            'x': list(lose_spans.values.squeeze()),
            'name': 'lost',
            'type': 'histogram',
            'nbins': 10,
            'histnorm': 'percent',
            'opacity': '0.75',
            'bingroup': 1,
            'marker': {'color': '#A83232'}
        }
    ]

    figure = {
        'data': graph_data,
        'layout': dict(
            barmode='overlay',
            margin={'l': 40, 'r': 15, 't': 10, 'b': 35},
            paper_bgcolor='#e8e8e8',
            font={'family': 'Dosis', 'size': 13},
            showlegend=False,
            yaxis={'title': 'Win/Lose (%)'},
            xaxis={'title': 'Time [h]'},
        )
    }
    return figure


def feature_figure(feature):
    label_dict = {
        'avg_profit': 'Average profit (%)',
        'expectancy': 'Expectancy',
        'avg_rrr': 'Average risk reward ratio',
        'win_rate': 'Win rate (%)',
        'avg_timespan': 'Average trade timespan (min)'
    }

    figure = {
        # Data is a list of dicts, each dict with x,y-data.
        'data': [
            {
                'x': user_data[feature].index,
                'y': user_data[feature].values,
                'name': 'feature',
                'line': dict(shape='linear', width=2,
                             color='#9DDCFA')
            },
        ],
        'layout': dict(
            # xaxis={'title': 'Date'},
            yaxis={'type': 'lin', 'title': label_dict[feature]},
            margin={'l': 50, 'r': 15, 't': 30, 'b': 35},
            height=410,
            legend={'x': 0.04, 'y': 0.96},
            paper_bgcolor='#e8e8e8',
            font={'family': 'Dosis', 'size': 13}
        )
    }
    return figure


def performance_indicators():
    total_loss = user_data['total_loss']
    total_gain = user_data['total_gain']
    avg_rrr = user_data['avg_rrr'][-1]
    avg_timespan = user_data['avg_timespan'][-1]/60

    trades = tl.read_trades(user_data['diary_file'], 'closed')
    size = trades['cap. share (%)'].mean()
    risk = trades['risk (%)'].mean()

    no_trades = len(trades)
    no_shorts = len(trades[trades['direction'] == 'SHORT'])
    no_longs = len(trades[trades['direction'] == 'LONG'])

    days_traded = trades['date'].apply(lambda x: datetime.date(x)).nunique()

    trades_per_day = no_trades/days_traded
    shortlongratio = no_shorts/no_longs

    entry = trades['entry']
    qty = trades['size']
    total_volume = sum(entry.mul(qty))
    max_loss = trades['P/L (%)'].min()
    max_win = trades['P/L (%)'].max()

    efficiency = (total_gain-total_loss)/total_volume*100

    return [html.H5('Additional Performance Indicators',
                    style={'margin-bottom': '20px'}),
            html.Div(
                [
                    html.Div(
                        [
                            html.Pre('Average trades per day:\t {:.1f}'.format(trades_per_day)),
                            html.Pre('Average timespan: \t \t {:.1f} h'.format(avg_timespan)),
                            html.Pre('Average r2r: \t \t \t {:.2f}'.format(avg_rrr)),
                            html.Pre('Average size: \t \t \t {:.0f} %'.format(size)),
                            html.Pre('Average risk: \t \t \t {:.1f} %'.format(risk)),
                            html.Pre('Short/long ratio: \t \t {:.1f}'.format(shortlongratio)),
                        ],
                        style={'width': '50%', 'display': 'flex', 'flex-flow': 'column',
                               'justify-content': 'space-between'}
                    ),
                    html.Div(
                        [
                            html.Pre('Efficiency: \t \t \t \t {:.2f} %'.format(efficiency)),
                            html.Pre('Total volume traded: \t {:.2f} $'.format(total_volume)),
                            html.Pre('Total Gain: \t \t \t \t {:.2f} $'.format(total_gain)),
                            html.Pre('Total Loss: \t \t \t \t {:.2f} $'.format(total_loss)),
                            html.Pre('Max loss:\t \t \t \t{:.2f} %'.format(max_loss)),
                            html.Pre('Max Win: \t \t \t \t  {:.2f} %'.format(max_win))
                        ],
                        style={'width': '50%', 'display': 'flex', 'flex-flow': 'column',
                               'justify-content': 'space-between'}
                    )
                ],
                style={'display': 'flex', 'height': '80%', 'overflow': 'auto'}
            )
            ]


def serve_layout():
    cap = user_data['capital'][-1]
    no_trades = len(user_data['capital']) - 1
    avg_profit = user_data['avg_profit'][-1]
    wl_rate = user_data['win_rate'][-1]
    expect = user_data['expectancy'][-1]

    return html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H5('Trades and Profits'),
                                    dcc.RadioItems(id='timespan', options=tl.spans, value='D',
                                                   style={'display': 'flex', 'margin-top': '0px',
                                                          'margin-bottom': '10px'},
                                                   labelStyle={'margin-right': '10px'}),
                                    dash_table.DataTable(
                                            id='summary_table',
                                            columns=summary_table_cols,
                                            data=[],
                                            style_table={
                                                # 'height': '400px',
                                                'overflow-y': 'scroll',
                                                'overflow-x': 'scroll'
                                            },
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
                                style={'margin-left': '0', 'padding-top': '5px'}
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
                                                    html.H5('{:.0f}%'.format(wl_rate*100), style={'margin': '0'}),
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
                                        style={'display': 'flex-inline', 'justify-content': 'space-between',
                                               'margin-right': '0'}
                                    ),
                                    html.Div(
                                        [
                                            html.H5('Capital Tracking', style={'text-align': 'center'}),
                                            html.Div(
                                                [
                                                    html.Div(
                                                        [
                                                            html.P('Monhtly profit target (%):',
                                                                   style={'display': 'inline', 'margin-right': '10px'}),
                                                            dcc.Input(id='monthly_target_profit', type='number',
                                                                      value=10,
                                                                      style={'margin': '0', 'width': '50px', 'display':
                                                                             'inline', 'height': '25px'})
                                                        ],
                                                    ),
                                                    html.Div(
                                                        [
                                                            dcc.RadioItems(id='y_opt', options=y_scale_options,
                                                                           value='lin',
                                                                           style={'display': 'flex',
                                                                                  'margin-top': '0px',
                                                                                  'margin-bottom': '0'},
                                                                           labelStyle={'margin-right': '10px'}),
                                                        ],
                                                        style={'margin-right': '15px'}
                                                    ),

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
                                                                target_capital_data(31, 10)
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
                                        style={'margin-left': '0', 'margin-right': '0', 'margin-top': '0',
                                               'padding-top': '5px'}
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
                                performance_indicators(),
                                className='pretty_container eight columns',
                                style={'margin-left': '0', 'margin-top': '0',
                                       'flex-flow': 'column', 'padding-top': '5px'}
                            ),
                            html.Div(
                                [
                                    html.H5('Short/Long Performance', style={'margin-bottom': '10px'}),
                                    dcc.Graph(
                                        id='shortlong_graph',
                                        figure=shortlong_figure()
                                    )
                                ],
                                className='pretty_container four columns',
                                style={'margin-top': '0', 'margin-right': '0', 'margin-left': '0', 'padding-top': '5px'}
                            )
                        ],
                        className='row flex-display',
                    ),
                    html.Div(
                        [

                            html.Div(
                                [
                                    html.H5('Trade Duration Performance'),
                                    dcc.Graph(
                                        id='timespan_graph',
                                        figure=timespan_figure()
                                    )
                                ],
                                className='pretty_container six columns',
                                style={'margin-left': '0', 'margin-top': '0', 'padding-top': '5px'}
                            ),
                            html.Div(
                                [
                                    html.H5('Feature Evolution'),
                                    dcc.Dropdown(id='feature_choice', options=features, value='win_rate'),
                                    dcc.Graph(
                                        id='feature_graph',
                                        figure=feature_figure('win_rate')
                                    )
                                ],
                                className='pretty_container six columns',
                                style={'margin-top': '0', 'margin-left': '0', 'margin-right': '0', 'padding-top': '5px'}
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

    n_days = 31  # HARDCODED CONSTANT !

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
            target_capital_data(n_days, target_profit)
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


@app.callback(Output('feature_graph', 'figure'),
              [Input('feature_choice', 'value')])
def update_feature_graph(feature):
    return feature_figure(feature)


@app.callback(Output('summary_table', 'data'),
              [Input('timespan', 'value')])
def update_summary_table(timedelta):
    trades = pd.read_excel(user_data['diary_file'], sheet_name='closed')
    # Throw away unnecessary columns and convert percents to demicals
    t = trades.loc[:, ['date', 'P/L (%)']]
    t['P/L (%)'] = t['P/L (%)'] / 100 + 1
    # Group dataframe by the appropriate frequency (Day, Week, Month)
    grouped = t.groupby(pd.Grouper(key='date', freq=timedelta))
    # Get total profit and number of trades:
    merged = grouped.prod()
    merged['trades'] = grouped.count()
    # Remove zero columns:
    # Note: decided not to do this yet

    # Convert back to percentages:
    merged['P/L (%)'] = round((merged['P/L (%)'] - 1) * 100, 2)
    # Add period descriptor:
    merged['period'] = merged.index.date
    if timedelta == 'D':
        pass
    elif timedelta == 'W':
        merged['period'] = merged['period'].apply(lambda x: datetime.strftime(x, '%Y W-%W'))
    elif timedelta == 'M':
        merged['period'] = merged['period'].apply(lambda x: datetime.strftime(x, '%b %Y'))
    # Flip dataFrame with newest data on top instead of bottom:
    merged = merged.iloc[::-1]

    return merged.to_dict(orient='records')
