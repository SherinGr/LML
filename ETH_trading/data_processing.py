# We will webscrape the data of ETH prices from cryptowat.ch
import cryptowatch as cw
import pandas as pd
import re
import os
import numpy as np

try:
    import cPickle as pickle
except:
    import pickle

from datetime import datetime


def create_empty_pickle_file(filename):
    file = open(filename, 'wb')
    pickle.dump([], file)
    file.close()

def get_data_file(time_frame):
    """ Function checks whether data file for the given timeframe already exists, and asks whether or not you want to
        use that data. Otherwise it makes a new file.

        :return filename in which data needs to be stored at the end of the session
    """

    # Get list of all pkl files in folder:
    all_files = os.listdir(path='.')
    pickle_files = [f for f in all_files if '.pkl' in f]
    # Check if any data for this time_frame exists:
    filename_regex = re.compile('data_' + time_frame.strip() + '_(\d+)' + '.pkl')
    matches = [f for f in pickle_files if re.match(filename_regex, f)]

    # Get the file numbers currently in use:
    file_nums = []
    for f in matches:
        num = re.search(filename_regex, f).group(1)
        file_nums.append(int(num))

    if matches:
        ans = input('Existing {} data found, do you want to add new candles to the old data? [y/n]\n'.format(
            time_frame.strip()))
        if ans in ['y', 'Y', 'yes', 'Yes']:
            # Retrieve the newest file (nf) and its modification date (mdate):
            mdates = [os.path.getmtime(f) for f in matches]
            nf_idx = np.argmax(np.array(mdates))
            nf = matches[nf_idx]
            nf_timestamp = np.max(np.array(mdates))
            nf_mdate = datetime.fromtimestamp(nf_timestamp)
            nf_mdate = nf_mdate.strftime('%Y-%m-%d')

            filename = nf
            print('Existing {} data loaded, last modified at {}.\n'.format(time_frame.strip(), nf_mdate) +
                  'New data will now be added if any is available.\n')
        else:
            print('OK, using new data only. Making new file to store the data.\n')
            new_nr = np.max(file_nums) + 1
            filename = 'data_' + time_frame.strip() + '_' + str(new_nr) + '.pkl'
            create_empty_pickle_file(filename)

    else:
        # First time creating a data file for this timeframe:
        filename = 'data_' + time_frame + '_0.pkl'
        create_empty_pickle_file(filename)
    return filename


def concatenate_batches(candle_set1, candle_set2):
    # TODO: check how many new candles there are and connect the sets

    # If sets do not connect (missing candle) ask user if OK to make new data file.
    return [], 0


def load_data(time_frame='4h'):
    """ This function webscrapes data of ETH prices on BINANCE. As it is now it returns 4h OHCL and volume data. This
    can be changed according to your needs.

    :return: candles_4h
    """

# Retrieve the file from which to load data and save it in (no data loaded if file is empty):
    data_filename = get_data_file(time_frame)
    data_file = open(data_filename, 'rb')
    old_candles = pickle.load(data_file)
    data_file.close()

# Get new data from the web:
    print('Fetching new data...')
    data = cw.markets.get("BINANCE:ETHUSDT", ohlc=True, periods=[time_frame])
    # The data is a list of lists, each element of the list is a list containing the following values:
    cols = data._legend
    # We will put the data in a pandas dataFrame:
    ref = 'data.of_' + time_frame
    new_candles = pd.DataFrame(eval(ref), columns=cols)

# Concatenate the old data and new data:
    if not old_candles.empty:
        all_candles, candles_added = concatenate_batches(old_candles, new_candles)
    else:
        all_candles = []
        candles_added = len(new_candles)

    print("Number of new {} candles added:".format(time_frame), candles_added)

    # We remove the volume_quote column, since it provides no additional information:
    # TODO: remove this if statement once concatenation works
    if not all_candles:
        candles = new_candles.drop(columns=['volume quote'])
    else:
        candles = all_candles.drop(columns=['volume quote'])

    return data_filename, candles


def save_data(filename, candle_data):  # *objects):
    # Start with dumping only the candles, expand later on if required.
    # You could also store things like trades or indicators here.
    data_file = open(filename, 'wb')
    pickle.dump(candle_data, data_file)

    data_file.close()





