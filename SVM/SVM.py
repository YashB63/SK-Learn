# SVM stands for Support Vector Machine, which is a popular machine learning algorithm used for classification and regression tasks. 
# The algorithm works by finding a hyperplane in a high-dimensional space that maximally separates the classes of the data.


# In SVM, the algorithm identifies the support vectors, which are the data points closest to the hyperplane. 
# The distance between the support vectors and the hyperplane is called the margin, and the SVM algorithm seeks to maximize this margin. 
# This margin maximization is achieved by minimizing the classification error, 
# subject to the constraint that the data points are correctly classified and lie on the correct side of the hyperplane.


# SVM can be used for both linear and nonlinear classification tasks, 
# through the use of kernel functions that transform the data into a higher-dimensional space, 
# where the data becomes more separable. Popular kernel functions include the linear kernel, polynomial kernel, 
# and radial basis function (RBF) kernel.


# SVM has some advantages, such as being effective in high-dimensional spaces 
# and for datasets with complex decision boundaries, 
# and providing a unique solution that does not depend on the initial conditions. 
# However, SVM can be computationally expensive for large datasets, and selecting the appropriate kernel function 
# and tuning the hyperparameters can be challenging.


# In addition to classification tasks, SVM can also be used for regression tasks, 
# where the algorithm finds a hyperplane that best fits the data while minimizing the error. 
# This variant of SVM is called Support Vector Regression (SVR).

from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm, metrics


iris = datasets.load_iris()

# split it in features and labels

X = iris.data
y = iris.target

classes = ['Iris Setosa', 'Iris Versicolour', 'Iris Virginica']
print(X.shape)
print(y.shape)

#hours of study vs good/bad grades

#10 different students
#train with 8
#predict with remaining two
#helps in determining the model accuracy

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)

model = svm.SVC()
model.fit(X_train, y_train)

print(model)

predictions = model.predict(X_test)
acc = metrics.accuracy_score(y_test, predictions)

print("Predictions: ", predictions)
print("Actual: ", y_test)
print("Accuracy: ", acc)


for i in range(len(predictions)):
    print(classes[predictions[i]])

