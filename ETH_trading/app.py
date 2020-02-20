# DASH application (GUI) for record keeping and inspecting performance.
# TODO:
#   1. Make a GUI to enter trade diary information, edit it and save it in a (csv?) file
#   2. Add capital tracking tab

# TODO:
#   1. Set standard size for input boxes narrower
#   2. Fix horizontal alignment of Pair box.

import pandas as pd
import numpy as np
import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from app_tabs import *

colors = {
    'background': '#313e5c',
    'text': '#fafafa'
}

app = dash.Dash(__name__)

app.layout = html.Div(
    [
        html.H1('- TRADING WORKBOOK -',
                style={'margin-top': '0px', 'color': colors['text'], 'text-align': 'center'}),
        html.H5("Good records are key to consistent profits",
                style={'margin-top': '0px', 'color': colors['text'], 'text-align': 'center'}),
        dcc.Tabs(id='app_tabs', value='open', vertical=False, children=[
            dcc.Tab(label='Open Trade', value='open'),
            dcc.Tab(label='Close Trade', value='close'),
            dcc.Tab(label='Performance', value='perf')
        ]),
        html.Div(id='tab_content')
    ]
)


@app.callback(Output('tab_content', 'children'), [Input('app_tabs', 'value')])
def render_content(tab):
    if tab == 'open':
        # OPEN NEW TRADES:
        return open_trade_tab
    elif tab == 'close':
        # RECORD OF OLD TRADES:
        return close_trade_tab
    elif tab == 'perf':
        # PERFORMANCE OF CAPITAL
        return html.H1('Performance')
    else:
        raise NameError('Tab with name "' + tab + '" does not exist')


if __name__ == '__main__':
    app.run_server(debug=True)
