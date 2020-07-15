import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Data.csv')

#create a matrix fro independent variables:
X = dataset.iloc[:, :-1].values #take all rows and :-1 cols(excluding last one)
y = dataset.iloc[:, 3].values

#split the dataset into training set and Test set:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


#feature scaling:
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

# print(X_train)
