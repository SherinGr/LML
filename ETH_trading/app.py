# DASH application (GUI) for record keeping and inspecting performance.
# TODO:
#   1. Make a GUI to enter trade diary information, edit it and save it in a (csv?) file
#   2. Add capital tracking tab

# TODO:
#   1. Set standard size for input boxes narrower
#   2. Fix horizontal alignment of Pair box.

import pandas as pd
import numpy as np

import datetime
from openpyxl import load_workbook

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
        dcc.Tabs(id='app_tabs', value='open', parent_className='custom_tabs', className='custom_tabs_container',
                 children=[
            dcc.Tab(label='Open Trade', value='open',
                    className='custom_tab', selected_className='custom_tab__selected'),
            dcc.Tab(label='Close Trade', value='close',
                    className='custom_tab', selected_className='custom_tab__selected'),
            dcc.Tab(label='Performance', value='perf',
                    className='custom_tab', selected_className='custom_tab__selected')
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


@app.callback(Output('open_table', 'data'), Input('button', 'n_clicks'),
              [State('pair', 'value'),
               State('entry', 'value'),
               State('size', 'value'),
               State('stop', 'value'),
               State('type', 'value'),
               State('direction', 'value'),
               State('confidence', 'value')]
              )
def submit_trade(clicks, pair, entry, size, stop, type, direction, confidence):
    # Add new trade at the top of the diary excel file:
    index = pd.DatetimeIndex([datetime.datetime.now()])
    # trade = pd.DataFrame(columns=open_trade_dict, index=index)
    trade = pd.Dataframe({
        'pair': pair, 'entry': entry, 'size': size, 'stop': stop, 'type': type, 'direction': direction, 'confidence':
            confidence
    }, index=index)

    with pd.ExcelWriter(path=diary, engine='openpyxl', datetime_format='DD-MM-YYYY hh:mm', mode='a') as \
            writer:
        # Open the file:
        writer.book = load_workbook(diary)
        # Copy existing sheets:
        writer.sheets = dict((ws.title, ws) for ws in writer.book.worksheets)
        # Add new trade on top of the existing data:
        writer.book['closed'].insert_rows(2)
        trade.to_excel(writer, sheet_name='closed', startrow=1, header=None, index_label='date')
        writer.close()


if __name__ == '__main__':
    app.run_server(debug=True)
