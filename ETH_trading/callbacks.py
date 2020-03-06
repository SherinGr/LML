from dash.dependencies import Input, Output, State
from app import app
from app_tabs import *


@app.callback(Output('entry', 'value'), Input('button', 'n_clicks'),
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

    data = get_open_trades(diary)
    return 0
