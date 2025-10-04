# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the CSV and separate features (Level) and target (Salary).

2.Fit a DecisionTreeRegressor with a chosen max_depth (e.g., 4).

3.Predict salaries on the same levels and compute MSE/R² if desired.

4.Visualize the trained tree to inspect its splits.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: R.Sairam
RegisterNumber: 25000694
*/
```
import pandas as pd

from sklearn.tree import DecisionTreeRegressor, plot_tree

from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt

df = pd.read_csv("Salary.csv")

X = df[["Level"]]

y = df["Salary"]

regressor = DecisionTreeRegressor(max_depth=4, random_state=42)

regressor.fit(X, y)

y_pred = regressor.predict(X)

print("MSE:",mean_squared_error(y, y_pred))

print("R squared:",r2_score(y, y_pred))

plt.figure(figsize=(15,8))

plot_tree(regressor, feature_names=["Level"], filled=True, fontsize=8)

plt.show()


## Output:
![Decision Tree Regressor Model for Predicting the Salary of the Employee](sam.png)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
