# DASH application (GUI) for record keeping and inspecting performance.
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from app import app
from tabs import open, close, performance

app.layout = html.Div(
    [
        html.Div(id="output-clientside"),
        html.H1('- TRADING WORKBOOK -',
                style={'margin-top': '0', 'margin-bottom': '0', 'color': '#fafafa', 'text-align': 'center'}),
        html.H5("Good records are the key to consistent profits",
                style={'margin-top': '0', 'color': '#fafafa', 'text-align': 'center'}),
        dcc.Tabs(id='app_tabs', value='open', parent_className='custom-tabs', className='custom-tabs-container',
                 children=[
                        dcc.Tab(label='Open Trade', value='open',
                                className='custom-tab', selected_className='custom-tab--selected',
                                style={'width': '33.3%'}, selected_style={'width': '66.7%'}
                                ),
                        dcc.Tab(label='Close Trade', value='close',
                                className='custom-tab', selected_className='custom-tab--selected',
                                style={'width': '33.3%'}, selected_style={'width': '66.7%'}
                                ),
                        dcc.Tab(label='Performance', value='perf',
                                className='custom-tab', selected_className='custom-tab--selected',
                                style={'margin-right': '0', 'width': '33.3%'}, selected_style={'margin-right': '0',
                                                                                               'width': '66.7%'}
                                )
                        ],
                 ),
        html.Div(id='tab_content')
    ]
)


@app.callback(Output('tab_content', 'children'), [Input('app_tabs', 'value')])
def render_content(tab):
    if tab == 'open':
        # OPEN NEW TRADES:
        return open.serve_layout()
    elif tab == 'close':
        # RECORD OF OLD TRADES:
        return close.serve_layout()
    elif tab == 'perf':
        # PERFORMANCE OF CAPITAL:
        return performance.serve_layout()
    else:
        raise NameError('Tab with name "' + tab + '" does not exist')


if __name__ == '__main__':
    app.run_server(debug=True)
