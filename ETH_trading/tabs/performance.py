import dash_core_components as dcc
import dash_html_components as html

layout = html.Div(
    [
        html.Div(
            [
                html.Div(
                    [

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
                                        html.H6(id='no_trades'),
                                        html.P('No. Trades')
                                    ],
                                    className='mini_container',
                                    style={'margin-left': '0', 'flex-grow': '1'}
                                ),
                                html.Div(
                                    [
                                        html.H6(id='avg_profit'),
                                        html.P('Avg. Profit')
                                    ],
                                    className='mini_container',
                                    style={'margin-left': '0', 'flex-grow': '1'}
                                ),
                                html.Div(
                                    [
                                        html.H6(id='wl_rate'),
                                        html.P('W/L Rate')
                                    ],
                                    className='mini_container',
                                    style={'margin-left': '0', 'flex-grow': '1'}
                                ),
                                html.Div(
                                    [
                                        html.H6(id='capital'),
                                        html.P('Capital ($)')
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
                                html.H1('GRAPH')
                            ],
                            className='pretty_container',
                            style={'margin-left': '0', 'margin-right': '0', 'margin-top': '0', 'display': 'flex'}
                        )
                    ],
                    className='eight columns',
                    style={}
                ),
             ],
            className='row flex-display'
        ),
        html.Div(
            [
                html.Div(
                    [
                      html.H1('SHORTLONG GRAPH')
                    ],
                    className='pretty_container four columns',
                    style={'margin-left': '0', 'margin-top': '0', 'margin-right': '0'}
                ),
                html.Div(
                    [
                      html.H1('TIMESPAN GRAPH')
                    ],
                    className='pretty_container four columns',
                    style={'margin-top': '0', 'margin-right': '0'}
                ),
                html.Div(
                    [
                      html.H1('MONTHLY STATS')
                    ],
                    className='pretty_container four columns',
                    style={'margin-top': '0', 'margin-right': '0'}
                ),
            ],
            className='row flex-display'
        )
    ]
)
