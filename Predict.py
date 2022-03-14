import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
from sklearn import preprocessing

# Reads dataset
diabetes = pd.read_csv('diabetes.csv')

# Removes duplicates and prints the number of rows and coloumns in the dataset
diabetes.drop_duplicates(inplace = True)
print(diabetes.head(8))
print(diabetes.shape)

# Converts the data into an array
dataset = diabetes.values

# Gets first eight columns of the dataset
X = dataset[:,0:8] 
# Gets the last column
Y = dataset[:,8] 

# Processes the dataset to bring feature values between 0 and 1 inclusive
scaler = preprocessing.MinMaxScaler()
X = scaler.fit_transform(X)

# Splits the training and testing dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state = 1)

class Perceptron:

  def __init__ (self):
    # w is weight and b is bias
    self.w = None
    self.b = None
     
  def model(self, x):
    return 1 if (np.dot(self.w, x) >= self.b) else 0
  
  # Predicts based on weight
  def prediction(self, X):
    Y = []
    for x in X:
      result = self.model(x)
      Y.append(result)
    return np.array(Y)
    
  def fit(self, X, Y, epochs = 1, lr = 1):
    self.w = np.ones(X.shape[1])
    self.b = 0
    accuracy = {}
    max_accuracy = 0
    weight_matrix = []
    for i in range(epochs):
      for x, y in zip(X, Y):
        y_pred = self.model(x)
        if y == 1 and y_pred == 0:
          self.w = self.w + lr * x
          self.b = self.b - lr * 1
        elif y == 0 and y_pred == 1:
          self.w = self.w - lr * x
          self.b = self.b + lr * 1
          
      weight_matrix.append(self.w)    
      accuracy[i] = accuracy_score(self.prediction(X), Y)
      if (accuracy[i] > max_accuracy):
        max_accuracy = accuracy[i]
        chkptw = self.w
        chkptb = self.b
    # Checkpoint (Save the weights and bias value)
    self.w = chkptw
    self.b = chkptb
        
    print(max_accuracy)
    # Plots the accuracy values over epochs
    plt.plot(accuracy.values())
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")
    plt.ylim([0, 1])
    plt.show()
    
    # Returns the weight matrix that contains weights over all epochs
    return np.array(weight_matrix)


perceptron = Perceptron()

# epochs = 10000 and lr = 0.3
weight_matrix = perceptron.fit(X_train, Y_train, 10000, 0.3)

# Makes predictions on test data
Y_predict_test = perceptron.prediction(X_test)

# Checks the accuracy of the model
print(accuracy_score(Y_predict_test, Y_test))
