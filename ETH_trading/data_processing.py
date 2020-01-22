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


def fetch_local_data(exchange, pair, time_frame):
    print('Fetching local data...')
    files = list_local_data_files(time_frame)

    newest_file = ''
    mdate = None
    if files:
        newest_file, mdate = last_modified_file(files)

    return newest_file, mdate


def fetch_web_data(exchange, pair, time_frame):
    """ Get new data from exchange:pair and save it in a pd.DataFrame"""
    print('Fetching new data from the web...')
    source = exchange.upper() + ':' + pair.upper()
    data = cw.markets.get(source, ohlc=True, periods=[time_frame])
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
        # Return newest dataset if no overlap
        clean_data = newest_dataset(candle_set1, candle_set2)
    else:
        joint_data = pd.concat([candle_set1, candle_set2], sort=False)  # important not to sort as to keep the right
        # candles when deleting duplicates

        index = joint_data.index
        is_duplicate = index.duplicated(keep='last')
        clean_data = joint_data[~is_duplicate]
        clean_data.sort_index()

    return clean_data, overlap


def merge_files(filename1, filename2):
    file1 = open(filename1, 'rb')
    file2 = open(filename2, 'rb')
    dataset1 = pickle.load(file1)
    dataset2 = pickle.load(file2)

    # Check that timeframes match:
    timeframes = ['1m', '15m', '5m', '30m', '1h', '2h', '4h', '6h', '12h', '1d', '3d', '1w', '1M']

    for tf in timeframes:  # this is not very robust, we just stop now after the first match.
        if tf in filename1:
            time_frame = tf.strip()
            break

    for tf in timeframes:
        if tf in filename2:
            time_frame2 = tf.strip()
            break

    if not time_frame == time_frame2:
        # In a later verstion also check to match exchange and pair. For now only Binance ETHUSDT data is assumed
        # if not is_data_compatible():
        raise Exception('Data sets are not compatible (different pair, exchange or timeframe.')

    merged_data, success = concatenate_batches(dataset1, dataset2)
    if not success:
        raise Exception('It is not possible to merge the provided files without a gap.')
    else:
        # Make new file and store the data in it:
        new_filename = create_new_data_file(time_frame)
        save_data(new_filename, merged_data)

    return new_filename


def merge_all_files(time_frame):
    files_list = list_local_data_files(time_frame)

    # Exit function if there's only one file:
    if len(files_list) == 1:
        print('Only one {} file present, nothing to merge.'.format(time_frame))
        return

    # Otherwise get the data from all the files:
    all_data = []
    for f in files_list:
        file = open(f, 'rb')
        data = pickle.load(file)
        print(str(type(data)) + f)
        if not data.empty:
            all_data = all_data + [data]

    # And merge all of it into a new file:
    while True:
        set1, set2 = all_data[:2]
        rest = all_data[2:]

        merged_set, success = concatenate_batches(set1, set2)
        if not success:
            raise Exception('It is not possible to merge the data without a gap.')
        elif rest:
            all_data = [merged_set] + rest
        elif not rest:
            # If the last two datasets have been merged, save data and stop:
            num_candles = len(merged_set)
            print("{} {} candles in the merged dataset.".format(num_candles, time_frame))
            new_filename = create_new_data_file(time_frame)
            save_data(new_filename, merged_set)
            break

    return new_filename


def format_data(candles):
    # Note that this function is sensitive to the column names

    # Make the timestamp the index:
    candles['DateTime'] = pd.to_datetime(candles['close timestamp'], unit='s')
    candles = candles.set_index(['DateTime'])
    candles = candles.drop(columns=['close timestamp'])

    # Remove the volume quote column, since it provides no additional information:
    candles = candles.drop(columns=['volume quote'])

    return candles


def load_data(exchange, pair, time_frame='1h'):
    """ This function webscrapes data of ETH prices on BINANCE. As it is now it returns 4h OHCL and volume data. This
    can be changed according to your needs.

    :return: candles
    """

    # Get data from the web:
    new_candles = fetch_web_data(exchange, pair, time_frame)
    new_candles = format_data(new_candles)

    # Check for local data:
    # TODO: Check that local data comes from the same exchange, pair and time_frame
    data_file, mdate = fetch_local_data(exchange, pair, time_frame)

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
                    print('No overlap between old and new data, will only use new data and store it in a new file.')
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

    print("{} {} candles in the current dataset.\n".format(num_candles, time_frame))

    return data_file, candles


def save_data(filename, candle_data):  # *objects):
    # Start with dumping only the candles, expand later on if required.
    # You could also store things like trades or indicators here.
    data_file = open(filename, 'wb')
    pickle.dump(candle_data, data_file)

    data_file.close()





