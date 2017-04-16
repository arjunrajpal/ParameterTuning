from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score,accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable

dataset = ['antV0', 'antV1', 'antV2', 'camelV0', 'camelV1', 'ivy', 'jeditV0', 'jeditV1', 'jeditV2', 'log4j', 'lucene', 'poiV0', 'poiV1', 'synapse', 'velocity', 'xercesV0', 'xercesV1']

precision_cart_original = [15, 54, 42, 30, 38, 21, 56, 32, 6, 95, 67, 65, 72, 50, 39, 17, 72]
precision_rf_original = [21, 67, 56, 28, 34, 23, 52, 32, 4, 95, 63, 67, 78, 60, 40, 28, 78]

precision_cart = []
precision_rf = []
precision_lr = []


# CART


def cart(a, b, c, d):
    ct = DecisionTreeClassifier(min_impurity_split=0.5, min_samples_split=2, min_samples_leaf=1)
    ct.fit(a, b)
    pred = ct.predict(c)
    # print "Predicted Matrix : "  + str(pred)

    p = precision_score(d, pred, average='weighted')
    p = accuracy_score(d, pred)

    # print p
    precision_cart.append(p * 100)


# Random Forest


def random_forest(a, b, c, d):
    rf = RandomForestClassifier(min_impurity_split=0.5, min_samples_split=2, min_samples_leaf=1, n_estimators=100)
    rf.fit(a, b)
    pred = rf.predict(c)
    # print "Predicted Matrix : "  + str(pred)

    p = precision_score(d, pred, average='weighted')
    # p = accuracy_score(d, pred)

    # print p
    precision_rf.append(p * 100)


# Logistic Regression


def logistic_regression(a, b, c, d):
    lr = LogisticRegression()
    lr.fit(a, b)
    pred = lr.predict(c)
    # print "Predicted Matrix : "  + str(pred)

    p = precision_score(d, pred, average='weighted')
    # p = accuracy_score(d, pred)

    # print p
    # print "\n"
    precision_lr.append(p * 100)


def perform_calculation(loc_train, loc_test):

    f1 = pd.read_csv(loc_train, delimiter=",")
    df1 = pd.DataFrame(f1)

    f2 = pd.read_csv(loc_test, delimiter=",")
    df2 = pd.DataFrame(f2)

    # Training Set
    X_Train_DF = df1.ix[:, 3:23]
    X_Train = X_Train_DF.values.astype(float)
    y_Train = np.asarray(list(df1["bug"]))
    # print X_Train
    # print Y_Train

    # Testing Set
    X_Test_DF = df2.ix[:, 3:23]
    X_Test = X_Test_DF.values.astype(float)
    y_Test = np.asarray(list(df2["bug"]))
    # print X_Test
    # print Y_Test

    cart(X_Train, y_Train, X_Test, y_Test)
    random_forest(X_Train, y_Train, X_Test, y_Test)
    logistic_regression(X_Train, y_Train, X_Test, y_Test)


# antV0
perform_calculation("/home/dushyant/Desktop/Github/ParameterTuning/combined_dataset_modified/antV0/1.csv", "/home/dushyant/Desktop/Github/ParameterTuning/combined_dataset_modified/antV0/3.csv")

# antV1
perform_calculation("/home/dushyant/Desktop/Github/ParameterTuning/combined_dataset_modified/antV1/2.csv", "/home/dushyant/Desktop/Github/ParameterTuning/combined_dataset_modified/antV1/4.csv")

# antV2
perform_calculation("/home/dushyant/Desktop/Github/ParameterTuning/combined_dataset_modified/antV2/3.csv", "/home/dushyant/Desktop/Github/ParameterTuning/combined_dataset_modified/antV2/5.csv")

# camelV0
perform_calculation("/home/dushyant/Desktop/Github/ParameterTuning/combined_dataset_modified/camelV0/1.csv", "/home/dushyant/Desktop/Github/ParameterTuning/combined_dataset_modified/camelV0/3.csv")

# camelV1
perform_calculation("/home/dushyant/Desktop/Github/ParameterTuning/combined_dataset_modified/camelV1/2.csv", "/home/dushyant/Desktop/Github/ParameterTuning/combined_dataset_modified/camelV1/4.csv")

# ivy
perform_calculation("/home/dushyant/Desktop/Github/ParameterTuning/combined_dataset_modified/ivy/1.csv", "/home/dushyant/Desktop/Github/ParameterTuning/combined_dataset_modified/ivy/3.csv")

# jeditV0
perform_calculation("/home/dushyant/Desktop/Github/ParameterTuning/combined_dataset_modified/jeditV0/1.csv", "/home/dushyant/Desktop/Github/ParameterTuning/combined_dataset_modified/jeditV0/3.csv")

# jeditV1
perform_calculation("/home/dushyant/Desktop/Github/ParameterTuning/combined_dataset_modified/jeditV1/2.csv", "/home/dushyant/Desktop/Github/ParameterTuning/combined_dataset_modified/jeditV1/4.csv")

# jeditV2
perform_calculation("/home/dushyant/Desktop/Github/ParameterTuning/combined_dataset_modified/jeditV2/3.csv", "/home/dushyant/Desktop/Github/ParameterTuning/combined_dataset_modified/jeditV2/5.csv")

# log4j
perform_calculation("/home/dushyant/Desktop/Github/ParameterTuning/combined_dataset_modified/log4j/1.csv", "/home/dushyant/Desktop/Github/ParameterTuning/combined_dataset_modified/log4j/3.csv")

# lucene
perform_calculation("/home/dushyant/Desktop/Github/ParameterTuning/combined_dataset_modified/lucene/1.csv", "/home/dushyant/Desktop/Github/ParameterTuning/combined_dataset_modified/lucene/3.csv")

# poiV0
perform_calculation("/home/dushyant/Desktop/Github/ParameterTuning/combined_dataset_modified/poiV0/1.csv", "/home/dushyant/Desktop/Github/ParameterTuning/combined_dataset_modified/poiV0/3.csv")

# poiV1
perform_calculation("/home/dushyant/Desktop/Github/ParameterTuning/combined_dataset_modified/poiV1/2.csv", "/home/dushyant/Desktop/Github/ParameterTuning/combined_dataset_modified/poiV1/4.csv")

# synapse
perform_calculation("/home/dushyant/Desktop/Github/ParameterTuning/combined_dataset_modified/synapse/1.csv", "/home/dushyant/Desktop/Github/ParameterTuning/combined_dataset_modified/synapse/3.csv")

# velocity
perform_calculation("/home/dushyant/Desktop/Github/ParameterTuning/combined_dataset_modified/velocity/1.csv", "/home/dushyant/Desktop/Github/ParameterTuning/combined_dataset_modified/velocity/3.csv")

# xercesV0
perform_calculation("/home/dushyant/Desktop/Github/ParameterTuning/combined_dataset_modified/xercesV0/1.csv", "/home/dushyant/Desktop/Github/ParameterTuning/combined_dataset_modified/xercesV0/3.csv")

# xercesV1
perform_calculation("/home/dushyant/Desktop/Github/ParameterTuning/combined_dataset_modified/xercesV1/2.csv", "/home/dushyant/Desktop/Github/ParameterTuning/combined_dataset_modified/xercesV1/4.csv")


sequence = ["Dataset", "Cart Precision (Paper)", "Cart (Our Model)", "Delta Cart", "Random Forest (Paper)", "Random Forest (Our Model)", "Delta Random Forest"]

t = PrettyTable(sequence)

for i in range(0,len(dataset)):
    t.add_row([dataset[i], precision_cart_original[i], precision_cart[i], abs(precision_cart_original[i] - precision_cart[i]), precision_rf_original[i], precision_rf[i], abs(precision_rf_original[i] - precision_rf[i])])


print t