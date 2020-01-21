# We will webscrape the data of ETH prices from cryptowat.ch
import cryptowatch as cw
import pandas as pd
import re
import os
import numpy as np
import pickle
import warnings

from datetime import datetime


# Functions for data file workflow:
def create_new_data_file(time_frame):
    matches = list_local_data_files(time_frame)
    if matches:
        file_nr = highest_file_nr(matches, time_frame) + 1
    else:
        file_nr = 0

    filename = 'data_' + time_frame.strip() + '_' + str(file_nr) + '.pkl'
    file = open(filename, 'wb')
    pickle.dump(pd.DataFrame(), file)
    file.close()

    return filename


def filename_regex(time_frame):
    return re.compile('data_' + time_frame.strip() + '_(\d+)' + '.pkl')


def list_local_data_files(time_frame):
    # Get list of all pkl files in folder:
    all_files = os.listdir(path='.')
    pickle_files = [f for f in all_files if '.pkl' in f]
    # Check if any data for this time_frame exists:
    regex = filename_regex(time_frame)
    matches = [f for f in pickle_files if re.match(regex, f)]

    return matches


def highest_file_nr(files_list, time_frame):
    """ Get the file numbers currently in use """
    regex = filename_regex(time_frame)

    file_nums = []
    for f in files_list:
        num = re.search(regex, f).group(1)
        file_nums.append(int(num))

    return max(file_nums)


def last_modified_file(files_list):
    """ From a list of files, return the last modified file and its modification date"""
    # Retrieve the newest file (nf) and its modification date (mdate):
    mdates = [os.path.getmtime(f) for f in files_list]
    nf_idx = np.argmax(np.array(mdates))
    nf = files_list[nf_idx]
    nf_timestamp = np.max(np.array(mdates))
    nf_mdate = datetime.fromtimestamp(nf_timestamp)
    nf_mdate = nf_mdate.strftime('%Y-%m-%d')

    return nf, nf_mdate


def fetch_local_data(time_frame):
    print('Fetching local data...')
    files = list_local_data_files(time_frame)

    newest_file = ''
    mdate = None
    if files:
        newest_file, mdate = last_modified_file(files)

    return newest_file, mdate


def fetch_web_data(time_frame):
    """ Get new data from BINANCE:ETHUSDT and save it in a pd.DataFrame"""
    # TODO: Allow for other markets and exchanges
    print('Fetching new data from the web...')
    data = cw.markets.get("BINANCE:ETHUSDT", ohlc=True, periods=[time_frame])
    # The data is a list of lists, each element of the list is a list containing the following values:
    cols = data._legend

    # We will put the data in a pandas dataFrame:
    ref = 'data.of_' + time_frame
    candle_df = pd.DataFrame(eval(ref), columns=cols)

    return candle_df


# Functions for batch concatenation:
def check_for_overlap(candle_set1, candle_set2):
    """ Check if there is overlap in the datasets """
    min_set1 = min(candle_set1.index)
    max_set1 = max(candle_set1.index)
    min_set2 = min(candle_set2.index)
    max_set2 = max(candle_set2.index)

    if not (min_set1 < max_set2 or max_set1 > min_set2):
        return False
    else:
        return True


def newest_dataset(candle_set1, candle_set2):
    max_set1 = max(candle_set1.index)
    max_set2 = max(candle_set2.index)

    if max_set1 > max_set2:
        return candle_set1
    else:
        return candle_set2


def concatenate_batches(candle_set1, candle_set2):

    overlap = check_for_overlap(candle_set1, candle_set2)
    if not overlap:
        completed_data = newest_dataset(candle_set1, candle_set2)
        print('No overlap between old and new data, will only use new data and store it in a new file.')
    else:
        joint_data = pd.concat([candle_set1, candle_set2])
        completed_data = joint_data.drop_duplicates()
        # TODO: DROP DUPLICATES DOES NOT WORK!?
        completed_data.sort_index()

    return completed_data, overlap


def merge_pickle_files():
    # TODO: write function to merge data in two pickle files if no gap present.
    pass


def format_data(candles):
    # Note that this function is sensitive to the column names

    # Make the timestamp the index:
    candles['DateTime'] = pd.to_datetime(candles['close timestamp'], unit='s')
    candles = candles.set_index(['DateTime'])
    candles = candles.drop(columns=['close timestamp'])

    # Remove the volume quote column, since it provides no additional information:
    candles = candles.drop(columns=['volume quote'])

    return candles


def load_data(time_frame='4h'):
    """ This function webscrapes data of ETH prices on BINANCE. As it is now it returns 4h OHCL and volume data. This
    can be changed according to your needs.

    :return: candles
    """

    # Get data from the web:
    new_candles = fetch_web_data(time_frame)
    new_candles = format_data(new_candles)

    # Check for local data:
    data_file, mdate = fetch_local_data(time_frame)

    if not data_file == '':
        ans = input('Existing {} data found, add new candles to the old data? [y/n]\n'.format(time_frame.strip()))

        if ans in ['y', 'Y', 'yes', 'Yes', '']:
            # Use newest file with old data in it:
            file = open(data_file, 'rb')
            old_candles = pickle.load(file)
            file.close()
            if old_candles.empty:
                warnings.warn('Old data file was empty, only new data available.')
                candles = new_candles
            else:
                print('Existing {} data loaded, last modified at {}.\n'.format(time_frame.strip(), mdate) +
                  'New data will now be added if any is available.\n')

                candles, success = concatenate_batches(old_candles, new_candles)
                if not success:
                    # Could not concatenate, save newest set of candles in a new file.
                    data_file = create_new_data_file(time_frame)
        else:
            # Create new file with higher file nr.
            print('OK, using new data only. Making new file to store the data: ')
            candles = new_candles
            data_file = create_new_data_file(time_frame)
    else:
        # Create first file for this timeframe
        print('Creating first {} data file: '.format(time_frame))
        candles = new_candles
        data_file = create_new_data_file(time_frame)

    num_candles = len(candles)

    print("{} {} candles in the current dataset.".format(num_candles, time_frame))

    return data_file, candles


def save_data(filename, candle_data):  # *objects):
    # Start with dumping only the candles, expand later on if required.
    # You could also store things like trades or indicators here.
    data_file = open(filename, 'wb')
    pickle.dump(candle_data, data_file)

    data_file.close()





