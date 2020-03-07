import dash
import shelve

app = dash.Dash(__name__)
app.config.suppress_callback_exceptions = True

diary = 'diary.xlsx'

d = shelve.open('user_profile', writeback=True)

# TODO: write a file that initializes these variables for one time use

capital_df = d['capital']
capital = capital_df.tail(1).value()


d.close()
