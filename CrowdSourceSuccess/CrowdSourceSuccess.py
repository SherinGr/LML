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
labels = np.ravel(data[['funded']].values)
data = data.drop('pledged', axis=1)

# transformation to get nice distribution:
#data.goal = 2/(np.exp(-data.goal/1)+1)-1

# Data cleaning:
data = data.drop(features[0], axis=1)

# Check if the converted feature makes sense, is there any difference with usd_pledged?
#conversion_difference = data[['converted_pledged_amount']].values - data[['usd_pledged']].values
# conclusion: there is a difference, use the converted as feature, drop this.


''' Lets inspect the data with some plots '''


f, axes = plt.subplots(1, 2)

#sns.distplot(data.usd_pledged)
#sns.distplot(data.backers_count)

#sns.scatterplot(x=data.goal, y=data.converted_pledged_amount, hue=data.funded, ax=axes[0])

#sns.countplot(data.currency, hue=data.funded)
# very skewed towards USD, maybe make binary (USD or not USD), but check labels first.

#jointplot
#facetgrid
#plt.show()

# make a jointplot of country vs currency, seems redundant information:

# make a pairplot of all continuous variables.

print(data.usd_pledged.describe())
print(data.backers_count.describe())
# both of these features need a transformation!

# seems the data is very skewed, apply minmax after capping off the data to maintain spread!

# To check categorical features:
print(data.category.unique())  # around 15
print(data.subcategory.unique())  # a lot! is it usefull?

""" The Classifier """

# We will make a tree based classifier
# First, we can already achieve 90% accuracy by using a naive Bayes classifier on conv_pledged vs goal.
from sklearn import svm
from sklearn.preprocessing import StandardScaler

clf = svm.LinearSVC()
scaler = StandardScaler()  # TODO: Use a better kernel and then a standard scaler.

data_cleaned = data[['converted_pledged_amount', 'goal']]
# SVM assumes: mean zero same order variance! Transform data!

scaler.fit_transform(data_cleaned)
data_transformed = scaler.transform(data_cleaned)

df = pd.DataFrame(data_transformed, columns=['converted_pledged_amount', 'goal'])
clf.fit(data_transformed, labels)

plot_sample = df.sample(100)
sample_labs = labels[plot_sample.index]
sns.scatterplot(x=plot_sample.goal,
                y=plot_sample.converted_pledged_amount,
                hue=sample_labs,
                ax=axes[0])

print(clf.score(data_cleaned, labels))


# Plot the classifier:
ax = plt.gca() # get current axes
xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 25),
                     np.linspace(ylim[0], ylim[1], 25))

z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
z = z.reshape(xx.shape)

plt.plot(xx,yy,z ax=ax)

