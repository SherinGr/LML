# DASH application (GUI) for record keeping and inspecting performance.

import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from app import app
from tabs import open, close, performance

app.layout = html.Div(
    [
        html.H1('- TRADING WORKBOOK -',
                style={'margin-top': '0', 'margin-bottom': '0', 'color': '#fafafa', 'text-align': 'center'}),
        # html.H5("Good records are key to consistent profits",
        #        style={'margin-top': '0px', 'color': colors['text'], 'text-align': 'center'}),
        dcc.Tabs(id='app_tabs', value='open', parent_className='custom-tabs', className='custom-tabs-container',
                 children=[
                        dcc.Tab(label='Open Trade', value='open',
                                className='custom-tab', selected_className='custom-tab--selected'),
                        dcc.Tab(label='Close Trade', value='close',
                                className='custom-tab', selected_className='custom-tab--selected'),
                        dcc.Tab(label='Performance', value='perf',
                                className='custom-tab', selected_className='custom-tab--selected')
                        ]
                 ),
        html.Div(id='tab_content')
    ]
)


@app.callback(Output('tab_content', 'children'), [Input('app_tabs', 'value')])
def render_content(tab):
    if tab == 'open':
        # OPEN NEW TRADES:
        return open.layout
    elif tab == 'close':
        # RECORD OF OLD TRADES:
        return close.layout
    elif tab == 'perf':
        # PERFORMANCE OF CAPITAL:
        return performance.layout
    else:
        raise NameError('Tab with name "' + tab + '" does not exist')


if __name__ == '__main__':
    app.run_server(debug=True)
