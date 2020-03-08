from binance.client import Client

import shelve

d = shelve.open('user_data')
keys = list(d.keys())
secret_key = d['secret_key']
api_key = d['api_key']

client = Client(api_key, secret_key)

ticker = client.get_symbol_ticker(symbol='ETHUSDT')
current_price = ticker['price']

info = client.get_margin_account()

d['capital'] = 49.349

d.close()

