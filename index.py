import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Salary_Data.csv')

#create a matrix fro independent variables:
X = dataset.iloc[:, :-1].values #take all rows and :-1 cols(excluding last one)
y = dataset.iloc[:, 1].values

#split the dataset into training set and Test set:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)


#feature scaling:
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""
# print(X_train)

#Fitting simple linear regression to the training set:
from sklearn.linear_model import LinearRegression
regressor = LinearRegression();
regressor.fit(X_train, y_train);


#predicting the test set Results:
y_pred = regressor.predict(X_test)


#visulaizing the training set result: + the predicted set...
plt.scatter(X_train, y_train, color='red')
plt.scatter(X_test, y_test, color='green')
plt.plot(X_train, regressor.predict(X_train), color='black')
plt.title('Salary vs Experience (Training Set = red; Predicted Set = green)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# print(X_train)