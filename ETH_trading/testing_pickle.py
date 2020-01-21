import pickle
import numpy as np

from data_processing import *

x = np.zeros((3, 3))

# Saving data:
data_file = open('test.pkl', 'wb')  # write binary

# for o in (x, y, z):
#     pickle.dump(o, data_file)

pickle.dump([], data_file)
data_file.close()

# Loading data:
file1 = 'data_15m_0.pkl'
load_file = open(file1, 'rb')  # read binary
data = pickle.load(load_file)

load_file.close()


# file2 = 'data_15m_4.pkl'
# load_file2 = open(file2, 'rb')
# data2 = pickle.load(load_file2)
#
# load_file2.close()
#
# new_filename = merge_files(file1, file2)
#
# new_file = open(new_filename, 'rb')
# merged_data = pickle.load(new_file)
# new_file.close()

filename = merge_all_files('15m')
file = open(filename, 'rb')
data = pickle.load(file)

idx = sum(np.diff(data.index))/len(data)  # check if all dt's are 15min