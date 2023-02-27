# logistic_regression.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# personal and educational purposes provided that (1) you do not distribute
# or publish solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UT Dallas, including a link to http://cs.utdallas.edu.
#
# This file is part of Programming Assignment 1 for CS6375: Machine Learning.
# Nikhilesh Prabhakar (nikhilesh.prabhakar@utdallas.edu),
# Athresh Karanam (athresh.karanam@utdallas.edu),
# Sriraam Natarajan (sriraam.natarajan@utdallas.edu),
#
#
# INSTRUCTIONS:
# ------------
# 1. This file contains a skeleton for implementing a simple version of the
# Logistic Regression algorithm. Insert your code into the various functions
# that have the comment "INSERT YOUR CODE HERE".
#
# 2. Do NOT modify the classes or functions that have the comment "DO NOT
# MODIFY THIS FUNCTION".
#
# 3. Do not modify the function headers for ANY of the functions.
#
# 4. You may add any other helper functions you feel you may need to print,
# visualize, test, or save the data and results.

'''
Shiva Kumar (sak220007)
Erik Hale   (emh170004)
'''

import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import pickle


class SimpleLogisiticRegression():
    """
    A simple Logisitc Regression Model which uses a fixed learning rate
    and Gradient Ascent to update the model weights
    """

    def __init__(self):
        self.w = []
        pass

    def initialize_weights(self, num_features):
        # DO NOT MODIFY THIS FUNCTION
        w = np.zeros((num_features))
        return w

    def compute_loss(self,  X, y):
        """
        Compute binary cross-entropy loss for given model weights, features, and label.
        :param w: model weights
        :param X: features
        :param y: label
        :return: loss   
        """
        # dot product of w,X
        val = np.dot(self.w, X)
        probability = self.sigmoid(val)
        loss = 0

        if y == 1:
            # When the sigmoid is smaller than what a float can hold but is still greather than 0,
            # Python will make this number 0.
            # To prevent having a np.log(0) which is infinity we have this test case
            if -((y)*np.log(probability)) != np.inf:
                loss = -((y)*np.log(probability))
            else:
                loss = 0

        else:
            # When the probability is 1 but y=0 the np.log will be infinity.
            # This is to prevent having a inf loss
            if -((1-y)*np.log(1-probability)) != np.inf:
                loss = -((1-y)*np.log(1-probability))
            else:
                loss = 0
        return loss

    def sigmoid(self, val):
        """
        Implement sigmoid function
        :param val: Input value (float or np.array)
        :return: sigmoid(Input value)
        """
        # Sigmoid function 1 / (1 + e^-x)
        return 1 / (1 + np.exp(-val))

    def gradient_ascent(self, w, X, y, lr):
        """
        Perform one step of gradient ascent to update current model weights. 
        :param w: model weights
        :param X: features
        :param y: label
        :param lr: learning rate
        Update the model weights
        """
        # array of y-yHat
        yHat = self.predict_example(self.w, X)
        # Array of the difference between the actual Y and the predicted y
        error = y - yHat
        # g represents the gradient vector
        g = w
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                g[j] += (error[i])*X[i][j]

        return self.w + lr*g

    # This helper function adds a column of 1's
    def columnof1s(self, X):
        # Creates an array that is the same number of rows as X and 1 column with one element: 1
        columns = np.ones((X.shape[0], 1))
        # combining X array with a columns array
        X = np.concatenate((X, columns), 1)
        return X

    def fit(self, X, y, lr=0.1, iters=100, recompute=True):
        """
        Main training loop that takes initial model weights and updates them using gradient descent
        :param w: model weights
        :param X: features
        :param y: label
        :param lr: learning rate
        :param recompute: Used to reinitialize weights to 0s. If false, it uses the existing weights Default True

        NOTE: Since we are using a single weight vector for gradient ascent and not using 
        a bias term we would need to append a column of 1's to the train set (X)

        """
        loss = []
        if (recompute):
            # Reinitialize the model weights
            # Creating a vector w with len(x) weights (0)
            self.w = self.initialize_weights(X.shape[1])
        # g will represent an initial gradient vector
        g = np.zeros(X.shape[1])
        for _ in range(iters):
            self.w = self.gradient_ascent(g, X, y, lr)
        # Returns the array of the loss values
        return self.w

    def predict_example(self, w, x):
        """
        Predicts the classification label for a single example x using the sigmoid function and model weights for a binary class example
        :param w: model weights
        :param x: example to predict
        :return: predicted label for x
        """
        yHats = []

        for i in range(x.shape[0]):
            # dot product between w and x[i]
            val = np.dot(self.w, x[i])
            # The predicted y-values is the sigmoid function
            yHat = self.sigmoid(val)

            if yHat > 0.5:
                yHat = 1
            else:
                yHat = 0
            yHats.append(yHat)
        return yHats

    def compute_error(self, y_true, y_pred):
        """
        Computes the average error between the true labels (y_true) and the predicted labels (y_pred)
        :param y_true: true label
        :param y_pred: predicted label
        :return: error rate = (1/n) * sum(y_true!=y_pred)
        """
        return (1/len(y_true))*sum(y_true != y_pred)


if __name__ == '__main__':

    # Load the training data
    M = np.genfromtxt('./data/monks-3.train', missing_values=0,
                      skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    # Load the test data
    M = np.genfromtxt('./data/monks-3.test', missing_values=0,
                      skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]

    lr = SimpleLogisiticRegression()
    # adds a column of 1's since we are not using a bias term and we are using a single vector for gradient ascent
    Xtrn = lr.columnof1s(Xtrn)
    Xtst = lr.columnof1s(Xtst)
    # Part 1) Compute Train and Test Errors for different number of iterations and learning rates
    for iter in [10, 100, 1000, 10000]:

        for a in [0.01, 0.1, 0.33]:
            # INSERT CODE HERE

            # Fit for Train Data set
            gradientAscent = lr.fit(Xtrn, ytrn, a, iter, True)
            # Call predict example
            predictTrainY = lr.predict_example(gradientAscent, Xtrn)

            # Call compute_error
            trainError = lr.compute_error(ytrn, predictTrainY)

            # Compute Test Errors
            predictTestY = lr.predict_example(gradientAscent, Xtst)
            # Call Test Errors
            testError = lr.compute_error(ytst, predictTestY)

            print("Train Error", trainError, "\n", "Iter",
                  iter, "\n", "Learning rate", a, "\n")
            print("Test Error", testError, "\n", "Iter",
                  iter, "\n", "Learning rate", a, "\n")

    # Part 2) Retrain Logistic Regression on the best parameters and store the model as a pickle file
    model = SimpleLogisiticRegression()
    # Apply the fit function to the model
    model.fit(Xtrn, ytrn, .01, 1000, True)
    # Code to store as pickle file
    # Students working on this assignement: Erik Hale, Shiva Kumar
    netid1 = 'emh170004'
    netid2 = 'sak220007'
    file_pi = open("emh170004_sak220007_model_1.obj", "wb")
    # file_pi = open('{}_model_1.obj', format(netid), 'wb')  #Use your NETID
    # print("Model:", model)
    pickle.dump(model, file_pi)

    # Part 3) Compare your model's performance to scikit-learn's LR model's default parameters
    # Initializing skikit's model
    skl_logistic_regression = LogisticRegression()

    # Training the model with training data
    skl_logistic_regression.fit(Xtrn, ytrn)

    # Predict the ys from the test set given the X features
    predictTestY = skl_logistic_regression.predict(Xtst)

    # accuracy of the model
    correctPercentage = skl_logistic_regression.score(Xtst, ytst)
    # If you want to see the accuracy
    print("Accuracy of Scikit learn: ", correctPercentage)

    # Finding the Error of the model (This is going to be more important for the assignment)
    # Error of the model is error = (1/n) * sum("y pred that are true" != "y pred")
    errorForSkl = lr.compute_error(ytst, predictTestY)
    # Fun fact: errorForSkl = 1 - correctPercentage
    print("Error of SciKit learn: ", errorForSkl)

    # Part 4) Plot curves on train and test loss for different learning rates. Using recompute=False might help
    for a in [0.01, 0.1, 0.33]:
        lr.fit(Xtrn, ytrn, lr=a, iters=1)
        # This array will contain the epoch numbers which is the x-axis
        Epoch = []
        # This array will contain the test loss for each iteration
        TrainLoss = []
        # This array will contain the test loss for each iteration
        TestLoss = []
        for i in range(10):
            lr.fit(Xtrn, ytrn, lr=a, iters=100, recompute=False)
            # For every row in Xtrn compute the train loss
            trainloss = 0
            for x in range(Xtrn.shape[0]):
                trainloss += lr.compute_loss(Xtrn[x], ytrn[x])
            # The formula for loss is (1/n) * (sum of losses)
            trainloss = trainloss / Xtrn.shape[0]
            TrainLoss.append(trainloss)

            # Computes Epoch Number
            epoch = 100*(i+1)
            Epoch.append(epoch)

            # For every row in Xtst compute the loss and add it to testLoss
            testLoss = 0
            for x in range(Xtrn.shape[0]):
                testLoss += lr.compute_loss(Xtst[x], ytst[x])
            # The formula for loss is (1/n) * (sum of losses)
            testLoss = testLoss / Xtst.shape[0]
            TestLoss.append(testLoss)
        # Labels for graph
        label = "Learning Rate " + str(a)
        plt.title(label)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        # Plot TrainLoss with red lines
        plt.plot(Epoch, TrainLoss, marker='o', color='r')
        # Plot TestLoss with blue lines
        plt.plot(Epoch, TestLoss, marker='o', color='b')
        plt.legend(["Training Loss", "Testing Loss"])
        plt.show()
