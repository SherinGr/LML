from binance.client import Client

import shelve

d = shelve.open('user_data')
keys = list(d.keys())

info = client.get_margin_account()


# Making a fake trade diary and capital evolution for performance page:


d['capital'] = 49.349

d.close()

