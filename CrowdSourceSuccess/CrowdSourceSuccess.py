import pandas as pd
import sklearn as skl
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# note that pandas and seaborn are compatible!

'''
    In this project we try to predict through selected features whether or not a kickstarter project will be a 
    success or not. 
'''

PATH = "./TrainData.csv"

# Data loading and pre-formatting:
data = pd.read_csv(PATH, skiprows=0, engine='python')
features = data.head(1).columns.tolist()
labels = data[['funded']].values
data = data.drop('pledged', axis=1)

# transformation to get nice distribution:
#data.goal = 2/(np.exp(-data.goal/1)+1)-1

# Data cleaning:
data = data.drop(features[0], axis=1)

# Check if the converted feature makes sense, is there any difference with usd_pledged?
conversion_difference = data[['converted_pledged_amount']].values - data[['usd_pledged']].values



''' Lets inspect the data with some plots '''


f, axes = plt.subplots(1, 2)

#sns.distplot(data.usd_pledged)
#sns.distplot(data.backers_count)

sns.scatterplot(x=data.goal, y=data.converted_pledged_amount, hue=data.funded, ax=axes[0])

#sns.countplot(data.currency, hue=data.funded)
# very skewed towards USD, maybe make binary (USD or not USD), but check labels first.

#jointplot
#facetgrid
plt.show()

# make a jointplot of country vs currency, seems redundant information:

#plt.show()

# make a pairplot of all continuous variables.

print(data.usd_pledged.describe())
print(data.backers_count.describe())
# both of these features need a transformation!

# seems the data is very skewed, apply minmax after capping off the data to maintain spread!

# To check categorical features:
print(data.category.unique()) # around 15
print(data.subcategory.unique()) # a lot! is it usefull?

""" The Classifier """

# We will make a tree based classifier
# First, we can already achieve 90% accuracy by using a naive Bayes classifier on conv_pledged vs goal.
#skl.naive_bayes()

