from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score,f1_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable

dataset = ['antV0', 'antV1', 'antV2', 'camelV0', 'camelV1', 'ivy', 'jeditV0', 'jeditV1', 'jeditV2', 'log4j', 'lucene', 'poiV0', 'poiV1', 'synapse', 'velocity', 'xercesV0', 'xercesV1']

precision_cart = []
precision_rf = []
precision_lr = []

fscore_cart = []
fscore_rf = []
fscore_lr = []

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

# CART


def cart(a, b, c, d):
    ct = DecisionTreeClassifier(min_impurity_split=0.5, min_samples_split=2, min_samples_leaf=1, max_features=None, max_depth=None)
    ct.fit(a, b)
    pred = ct.predict(c)
    # print "Predicted Matrix : "  + str(pred)

    p = precision_score(d, pred, average='weighted')
    # p = accuracy_score(d, pred)

    # print p
    precision_cart.append(p * 100)

    f = f1_score(d, pred, average='weighted')
    fscore_cart.append(f * 100)


# Random Forest


def random_forest(a, b, c, d):
    rf = RandomForestClassifier(min_impurity_split=0.5, min_samples_split=2, min_samples_leaf=1, n_estimators=100, max_features=None, max_leaf_nodes=None)
    rf.fit(a, b)
    pred = rf.predict(c)
    # print "Predicted Matrix : "  + str(pred)

    p = precision_score(d, pred, average='weighted')
    # p = accuracy_score(d, pred)

    # print p
    precision_rf.append(p * 100)

    f = f1_score(d, pred, average='weighted')
    fscore_rf.append(f * 100)

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

    f = f1_score(d, pred, average='weighted')
    fscore_lr.append(f * 100)


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


def calculate_all():

    # antV0
    perform_calculation("../combined_dataset_modified/antV0/1.csv", "../combined_dataset_modified/antV0/3.csv")

    # antV1
    perform_calculation("../combined_dataset_modified/antV1/2.csv", "../combined_dataset_modified/antV1/4.csv")

    # antV2
    perform_calculation("../combined_dataset_modified/antV2/3.csv", "../combined_dataset_modified/antV2/5.csv")

    # camelV0
    perform_calculation("../combined_dataset_modified/camelV0/1.csv", "../combined_dataset_modified/camelV0/3.csv")

    # camelV1
    perform_calculation("../combined_dataset_modified/camelV1/2.csv", "../combined_dataset_modified/camelV1/4.csv")

    # ivy
    perform_calculation("../combined_dataset_modified/ivy/1.csv", "../combined_dataset_modified/ivy/3.csv")

    # jeditV0
    perform_calculation("../combined_dataset_modified/jeditV0/1.csv", "../combined_dataset_modified/jeditV0/3.csv")

    # jeditV1
    perform_calculation("../combined_dataset_modified/jeditV1/2.csv", "../combined_dataset_modified/jeditV1/4.csv")

    # jeditV2
    perform_calculation("../combined_dataset_modified/jeditV2/3.csv", "../combined_dataset_modified/jeditV2/5.csv")

    # log4j
    perform_calculation("../combined_dataset_modified/log4j/1.csv", "../combined_dataset_modified/log4j/3.csv")

    # lucene
    perform_calculation("../combined_dataset_modified/lucene/1.csv", "../combined_dataset_modified/lucene/3.csv")

    # poiV0
    perform_calculation("../combined_dataset_modified/poiV0/1.csv", "../combined_dataset_modified/poiV0/3.csv")

    # poiV1
    perform_calculation("../combined_dataset_modified/poiV1/2.csv", "../combined_dataset_modified/poiV1/4.csv")

    # synapse
    perform_calculation("../combined_dataset_modified/synapse/1.csv", "../combined_dataset_modified/synapse/3.csv")

    # velocity
    perform_calculation("../combined_dataset_modified/velocity/1.csv", "../combined_dataset_modified/velocity/3.csv")

    # xercesV0
    perform_calculation("../combined_dataset_modified/xercesV0/1.csv", "../combined_dataset_modified/xercesV0/3.csv")

    # xercesV1
    perform_calculation("../combined_dataset_modified/xercesV1/2.csv", "../combined_dataset_modified/xercesV1/4.csv")

    # Print Precision table for Cart, Random Forest and Logistic Regression
    sequence_precision = ["Dataset", "Logistic Regression", "Cart", "Random Forest"]

    t_precision = PrettyTable(sequence_precision)

    for i in range(0, len(dataset)):
        t_precision.add_row([dataset[i],precision_lr[i], precision_cart[i], precision_rf[i]])

    print color.BOLD + color.CYAN + "\nTable for Precision comparison for Untuned Learner" + color.END
    print str(t_precision) + "\n"

    # Print fscore table for Cart, Random Forest
    sequence_fscore = ["Dataset", "Logistic Regression", "Cart", "Random Forest"]

    t_fscore = PrettyTable(sequence_fscore)

    for i in range(0, len(dataset)):
        t_fscore.add_row([dataset[i], fscore_lr[i], fscore_cart[i], fscore_rf[i]])

    print color.BOLD + color.CYAN + "\nTable for fscore comparison for Untuned Learner" + color.END
    print t_fscore

    return precision_cart,precision_rf, precision_lr,fscore_cart,fscore_rf,fscore_lr

if __name__ == "__main__":
    calculate_all()

