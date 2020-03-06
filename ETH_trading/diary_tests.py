import pandas as pd
import datetime
from openpyxl import load_workbook

record_file_path = "trading_diary.xlsx"
# writer = pd.ExcelWriter(path=record_file_path, engine='openpyxl', datetime_format='DD-MM-YYY HH:MM', mode='a')

# Define a trade dataFrame standard:
cols = ['pair', 'size', 'entry', 'exit', 'initial stop', 'P/L (USDT)', 'P/L (%)', 'pre capital',
        'risk', 'rrr', 'direction', 'type', 'confidence', 'note']

index = pd.DatetimeIndex([datetime.datetime.now()])

trade = pd.DataFrame(columns=cols, index=index)
trade['pair'] = 'ETHUSDT'
trade['size'] = 1.5
trade['entry'] = 184.04
trade['initial stop'] = 180.32


def add_trade(diary, trade):
    # Add new trade at the top of the diary excel file.
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


def get_latest_trade(diary):
    return pd.read_excel(diary, nrows=1, sheet_name='open')


def edit_trade(diary, id):
    # user selected an open trade and wants to edit it
    pass


def close_trade(diary, id):
    # User selects an open trade and wants to close it (edit it).
    pass


add_trade(record_file_path, trade)
new = get_latest_trade(record_file_path)
