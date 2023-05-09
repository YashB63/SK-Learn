# K-Nearest Neighbors (KNN) is a popular machine learning algorithm used for classification and regression tasks. 
# In KNN, the algorithm classifies new data points based on the majority class of the K closest labeled data points in the training set.
# The algorithm is called "K-nearest neighbors" because it finds the K closest neighbors to the new data point, 
# based on some distance metric, such as Euclidean distance or Manhattan distance.



# Here's an example of how the KNN algorithm works for classification:

# 1. Choose a value for K.
# 2. For each data point in the test set, calculate the distance to all data points in the training set.
# 3. Select the K data points in the training set that are closest to the test data point based on the distance metric.
# 4. Determine the majority class of the K data points.
# 5. Assign the test data point to the majority class.
# KNN is a non-parametric algorithm, meaning it does not make assumptions about the underlying distribution of the data. 
# It is also a lazy learning algorithm, meaning it does not learn a model from the training data but instead stores the 
# training data and performs computations at the time of prediction.



# KNN has some advantages, such as being simple to understand and implement, and often performing well on small datasets 
# or datasets with complex decision boundaries. However, KNN can also be computationally expensive for large datasets 
# and can suffer from the "curse of dimensionality" when the number of features is high.




import numpy as np
import pandas as pd
from sklearn import neighbors, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('SKLEARN\car_evaluation.csv')
print(data.head())

X = data[[
    'buying',
    'maint',
    'safety'
]].values

y = data[['class']]

print(X, y)

#converting the data of X
Le = LabelEncoder()
for i in range(len(X[0])):
    X[:, i] = Le.fit_transform(X[:, i])

print(X)


#In scikit-learn (sklearn), the LabelEncoder is a utility class that 
#can be used to encode categorical labels (i.e., non-numeric labels) into numerical labels.

#The LabelEncoder works by assigning a unique integer value to each unique category in the input data. 
#This can be useful for machine learning algorithms that require numeric input, such as decision trees or neural networks.

#maan lo ek aisa example ha:
#encoder.fit(['cat', 'dog', 'fish', 'dog', 'cat'])

# transform the input data using the encoder
# encoded_labels = encoder.transform(['cat', 'dog', 'fish', 'dog', 'cat'])

# print(encoded_labels) 

# Ye aayega Output -> [0 1 2 1 0]
#yaha kar kya raha hai ki cat ko 0 value assign ki jaa rhi hai
#dog ko 1 and fish ko 2
#simply koi bhi non numeric cheez ko numeric cheez mey convert karna

#converting the data of y:
label_maping = {
    'unacc':0,
    'acc':1,
    'good':2,
    'vgood':3
}
y['class'] = y['class'].map(label_maping)

#Scikit-learn (sklearn) does not have a built-in feature called "label mapping." 
#However, it is possible to create a label mapping using a combination of sklearn utilities, 
#such as LabelEncoder and DictVectorizer.

# In machine learning, label mapping is a process of transforming a set of labels into a new set of labels. 
# This can be useful when you have multiple labels that refer to the same thing, or when you want to group similar labels together.


# labels = ['cat', 'dog', 'fish', 'dog', 'cat', 'fish']
# # create the label mapping
# mapping = {'cat': 'pet', 'dog': 'pet', 'fish': 'wildlife'}

y = np.array(y)
print(y)

#create model

knn = neighbors.KNeighborsClassifier(n_neighbors=25, weights='uniform')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

knn.fit(X_train, y_train)

prediction = knn.predict(X_test)

accuracy = metrics.accuracy_score(y_test, prediction)

print("predictions: ", prediction)
print("Accuracy: ", accuracy)

print("actual value:", y[20])
print("predicted value: ", knn.predict(X)[20])