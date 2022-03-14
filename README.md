# Predicting-Diabetes-using-Neural-Network
Predicting Diabetes using Neural Network 

We build a neural network in Python to predict diabetes using the Pima Diabetes Dataset. The Pima Diabetes Dataset which has 8 numerical predictors and a binary outcome:
1.	Number of times pregnant
2.	Plasma glucose concentration a 2 hours in an oral glucose tolerance test
3.	Diastolic blood pressure (mm Hg)
4.	Triceps skin fold thickness (mm)
5.	2-Hour serum insulin (mu U/ml)
6.	Body mass index (weight in kg/(height in m)^2)
7.	Diabetes pedigree function
8.	Age (years)
9.	Class variable (0 or 1)
We design your neural network based on the perceptron model. 


A perceptron model is designed using basic libraries like numpy, pandas and scikit-learn to process, train and compile the dataset and matplotlib to plot the accuracy value over the 10000 training cycles, i.e. epochs.
The diabetes.csv file is imported and converted in an array after removing all duplicate entries.
Then the dataset is split into two. X contains all the feature data (row 1 to 8) and Y contains the target data (0 or 1, results in row 9). X is processed using the min-max scaler method to contain values between 0 and 1.
The data is split into two- 70% training and 30% testing. At this point we have X_train, Y_train and X_test, Y_test.
These sets are used to train and test a Neural Network designed to predict the results based on the perceptron model with zero hidden layers.
The perceptron class has a prediction function, a model function and a fit function. Prediction function uses the model to predict result Y based on the weight and returns it. The fit function evaluates and models the predicted result, training the neural network by performing several iterations (epochs). It
 
compares the predicted result to the training result dataset and sets the weight and bias accordingly. Based on the accuracy score, it marks the checkpoints (weight and bias with the highest accuracy score) and returns the weight matrix that contains weights over all epochs.
After training the Neural Network on the training dataset, we make predictions on the test data and print the accuracy score attained.
Matplotlib is used to plot the accuracy values over all the epochs.
