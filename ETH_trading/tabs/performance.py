import dash_core_components as dcc
import dash_html_components as html

layout = html.Div(
    [
        html.Div(children=[
            dcc.Tabs(id='numeric_stats', children=[
                dcc.Tab(label='Monthly Profits', children=['Testing text here'],
                        className='custom-tab', selected_className='custom-tab--selected',
                        style={'border-top-left-radius': '4px'},
                        selected_style={'border-top-left-radius': '4px'}),
                dcc.Tab(label='Something', children=['More text here'],
                        className='custom-tab', selected_className='custom-tab--selected',
                        style={'border-top-right-radius': '4px'},
                        selected_style={'border-top-right-radius': '4px',
                                        'background_color': 'coral'})
                        ]
                     )
                ],
            className='pretty_container four columns',
            style={'padding': '0', 'margin-left': '0', 'margin-top': '0', 'display': 'inline-block'}
            ),
        html.Div(
            [
              html.H1('GRAPH')
            ],
            className='pretty_container',
            style={'margin-right': '0', 'display': 'flex'}
        )
    ]
)
