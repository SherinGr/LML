# Why use pandas?
# Good at handling missing data
# Easy data manipulation
# Time series tools

# Series (one-dimension) or Data Frame (2 dimensions)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Series examples:
pd.Series([1., 2., 3.])
pd.Series([1., 2., 3.], index=['a', 'b', 'c'])
pd.Series([1., 2., np.nan])  # missing data

# Data frame examples:
h = [[1, 2], [3, 4]] # list of lists! Not array!
df_h = pd.DataFrame(h)
print('Data Frame: \n', df_h)
# To numpy array:
df_h_np = np.array(df_h)
print('Numpy array:\n', df_h_np)

# Dictionary to dataframe:
dic = {'Name': ['John', 'Smith'],
       'Age': [30, 40]}

df_dic = pd.DataFrame(data=dic)

# Range data: pd.date_range(date, period, frequency)
dates_h = pd.date_range('20190924', periods=6, freq='h')

# Inspecting data:
rand_n = np.random.randn(6, 4)
df = pd.DataFrame(rand_n, index=dates_h,
                  columns=list('ABCD')
                  )
print(df.head(4))
print(df.tail(4))

info = df.describe()
slice_col = df[['B', 'D']]
slice_row = df[0:3]
slice_loc = df.loc[:, ['B', 'D']]  # row and col by name
slice_iloc = df.iloc[[0, 2], [1, 3]]
print(slice_iloc)

# Manipulate data:
cleaned = df.drop(columns=['A', 'C'])

# concatenate dataframes:
df1 = pd.DataFrame({'Name': ['John', 'Smith', 'Paul'],
                    'Age': ['25', '30', '50']},
                   )
df2 = pd.DataFrame({'Name': ['Adam', 'Smith'],
                    'Age': ['26', '11']},
                   )
df_conc = pd.concat([df1, df2])

print(df_conc)

# delete duplicates (choose column name):
cleaned2 = df_conc.drop_duplicates('Name')
# sort rows:
sort = cleaned2.sort_values('Age')

# rename a column:
sort.rename(columns={'Name': 'Surname'})
# renumber the rows
sort = sort.reset_index(drop=True)
print(sort)

""" Working with real data, the adult dataset"""
# Loading a CSV file:
COLS = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
        'marital_stat', 'occupation', 'relationship', 'race', 'sex',
        'capital_gain', 'capital_loss', 'hours_week', 'native_country',
        'label']

PATH = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

data = pd.read_csv(PATH, names=COLS, index_col=False)

# grouping the data:
grouped = data.groupby('label')
summarized = grouped.mean()
minage = grouped['age'].min()

# more complex grouping:
test = data.groupby(['label', 'marital_stat'])['capital_gain'].mean()

# make a plot of the data:
df_plot = test.unstack()
df_plot.plot.bar()
plt.show()