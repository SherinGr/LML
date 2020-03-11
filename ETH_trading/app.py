import dash
import shelve
import binance
import shelve
from binance.client import Client

user_data = shelve.open('user_data', writeback=True)
# TODO: Open and close the shelf in functions that use it!!!

api_key = user_data['api_key']
secret_key = user_data['secret_key']
client = Client(api_key, secret_key)

app = dash.Dash(__name__)
app.config.suppress_callback_exceptions = True

