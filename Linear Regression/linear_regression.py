#What is Regression?

# Regression analysis is a form of predictive modelling technique 
#which investigates the relationship between a dependent and independent variable.

#Uses of Regression:

# - Determining the strength of predictors
# - Forecasting an effect, and
# - Trend Forecasting


#Linear Regression vs Logistic Regression

#Linear Regression:

#Core Concept: The Data is modelled using a straight line.
#Used with: Continuous variable
#Output/Prediction: Value of the variable
#Accuracy and Goodness of fit: Measured by loss, R squared, Adjusted R squared, etc.

#Logistic Regression:

#Core Concept: The probability of some obtained event is represented as a linear function of a combination of predictor variables.
#Used With: Categorical Variables
#Output/Prediction: Probability of occurence of event
#Accuracy and Goodness of fit: Accuracy, Precision, Recall, F1 Score, ROC curve, Confusion Matrix, etc.

#Linear Regression Selection Criteria:

# - Classification and Regression Capabilities
# - Data Quality
# - Computational Complexity
# - Comprehensible and Transparent

#Where is Linear Regression Used?

# - Evaluating Trends and Sales Estimates
# - Analyzing the  Impact of Price Changes
# - Assessment of risk in financial services and insurance domain

#Goodness of fit

#What is R-Square?

#R square value is a statistical measure of how close the data are to the fitted regression line
# It is also known as coefficient of determination, or the coefficient of multiple determination.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
plt.rcParams['figure.figsize'] = (20.0, 10.0)

#Reading Data
data = pd.read_csv('headbrain.csv')
print(data.shape)
print(data.head())

#Collecting X and Y
X = data['Head Size(cm^3)'].values
Y = data['Brain Weight(grams)'].values

# Mean X and Y
mean_x = np.mean(X)
mean_y = np.mean(Y)

#Total number of values
m = len(X)

#Using the formula to calculate b1 and b2
numer = 0
denom = 0
for i in range(m):
    numer += (X[i] - mean_x) * (Y[i] - mean_y)
    denom += (X[i] - mean_x) ** 2
b1 = numer / denom
b0 = mean_y - (b1 * mean_x)

#Print Coefficients
print(b1, b0)

#Plotting Values and Regression Line
max_x = np.max(X) + 100
min_x = np.min(X) - 100

#Calculating line values x and y
x = np.linspace(min_x, max_x, 1000)
y = b0 + b1 * x

#Plotting Line
plt.plot(x, y, color='#FF0000', label='Regression Line')

#Plotting Scatter Points
plt.scatter(X, Y, c='#00FFFF', label='Scatter Plot')

plt.xlabel('Head Size in cm3')
plt.ylabel('Brain Weight in grams')
plt.legend()
plt.show()

ss_t = 0
ss_r = 0
for i in range(m):
    y_pred = b0 + b1 * X[i]
    ss_t += (Y[i] - mean_y) ** 2
r2 = 1 - (ss_r/ss_t)
print(r2)

# ------ linear Regression using Sklearn template-------------

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#cannot use Rank1 matrix in scikit learn
X = X.reshape(m,1)

#creating Model
reg = LinearRegression()

#Fitting training Data
reg = reg.fit(X, y)

#Y prediction
Y_pred = reg.predict(X)

#calculating R2 Score
r2_score = reg.score(X, Y)
print(r2_score)