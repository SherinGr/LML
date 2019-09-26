# Components to remember:
#
# Linear model
# Logistic function (sigmoid)
# Confusion matrix (class imbalance analysis)
# Precision (fraction of correctly predicted positives)
# Sensitivity (fraction of all true positives detected)

import tensorflow as tf
import pandas as pd

# We will classify +/-50K income

# Data loading:
COLUMNS = ['age','workclass', 'fnlwgt', 'education', 'education_num', 'marital',
           'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',
           'hours_week', 'native_country', 'label']
PATH = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
PATH_test = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"

train = pd.read_csv(PATH, skipinitialspace=True,
                    names=COLUMNS,
                    index_col=False)

test = pd.read_csv(PATH_test, skipinitialspace=True,
                   skiprows=1,
                   names=COLUMNS,
                   index_col=False)

# Make the income data a boolean (0 or 1):
label = {'<=50K': 0, '>50K': 1}
train_lab = [label[item] for item in train.label]
