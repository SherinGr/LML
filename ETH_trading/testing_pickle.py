import pickle
import numpy as np

x = np.zeros((3, 3))

# Saving data:
data_file = open('test.pkl', 'wb')  # write binary

# for o in (x, y, z):
#     pickle.dump(o, data_file)

pickle.dump([], data_file)
data_file.close()

# Loading data:
load_file = open('test.pkl', 'rb')  # read binary

data = pickle.load(load_file)

load_file.close()

if True:
    print("True")
else:
    print("False")