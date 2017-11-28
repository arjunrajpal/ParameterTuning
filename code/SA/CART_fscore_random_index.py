from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
import numpy as np
import math
from data import data
from random import random, randint

# Runs CART algorithm on the dataset using parameters present in candidate

global_data = []

def cart(a, b, c, d, tunings):
    # print "\nCandidate in Cart : " + str(candidate)
    # # value_threshold = nump.random.uniform(0.01, 1)
    # # print "Threshold :" + str(value_threshold)
    # print "Threshold :" + str(candidate["tunings"][0])
    # print "Max feature : " + str(candidate["tunings"][1])
    # print "Min sample split : " + str(candidate["tunings"][2])
    # print "Min samples leaf : " + str(candidate["tunings"][3])
    # print "Max depth : " + str(candidate["tunings"][4])

    ct = DecisionTreeClassifier(max_depth=tunings[4], min_samples_split=tunings[2], min_samples_leaf=tunings[3],
                                max_features=tunings[1], min_impurity_split=tunings[0])
    ct.fit(a, b)
    pred = ct.predict(c)
    # print "Predicted Matrix : "  + str(pred)
    p = f1_score(d, pred, average='weighted')

    return p


# Retrieves the appropriate dataset from readDataset function present in data.py and passes
# the Training and Testing dataset to CART function


def score(tunings, dataset):

    df1, df2 = data.readDataset(dataset)

    # Training Set
    X_Train_DF = df1.ix[:, 3:23]
    X_Train = X_Train_DF.values.astype(float)
    Y_Train = np.asarray(list(df1["bug"]))
    # print X_Train
    # print Y_Train

    # Testing Set
    X_Test_DF = df2.ix[:, 3:23]
    X_Test = X_Test_DF.values.astype(float)
    Y_Test = np.asarray(list(df2["bug"]))
    # print X_Test
    # print Y_Test

    p = cart(X_Train, Y_Train, X_Test, Y_Test, tunings)

    return p


def score_test(tunings, dataset):

    df1, df2 = data.readDataset_for_testing(dataset)

    # Training Set
    X_Train_DF = df1.ix[:, 3:23]
    X_Train = X_Train_DF.values.astype(float)
    Y_Train = np.asarray(list(df1["bug"]))
    # print X_Train
    # print Y_Train

    # Testing Set
    X_Test_DF = df2.ix[:, 3:23]
    X_Test = X_Test_DF.values.astype(float)
    Y_Test = np.asarray(list(df2["bug"]))
    # print X_Test
    # print Y_Test

    p = cart(X_Train, Y_Train, X_Test, Y_Test, tunings)

    return p


def population_initialisation(algoParameters):

    # List for the initial/old tuning values
    old_tunings = []

    # Population initialisation using random values within the parametric ranges

    threshold = np.random.uniform(algoParameters[0]['low'], algoParameters[0]['high'])
    old_tunings.append(threshold)
    # print threshold

    max_feature = int(np.random.uniform(algoParameters[1]['low'], algoParameters[1]['high']))
    old_tunings.append(max_feature)
    # print max_feature

    min_sample_split = int(np.random.uniform(algoParameters[2]['low'], algoParameters[2]['high']))
    old_tunings.append(min_sample_split)
    # print min_sample_split

    min_samples_leaf = int(np.random.uniform(algoParameters[3]['low'], algoParameters[3]['high']))
    old_tunings.append(min_samples_leaf)
    # print min_samples_leaf

    max_depth = int(np.random.uniform(algoParameters[4]['low'], algoParameters[4]['high']))
    old_tunings.append(max_depth)
    # print max_depth

    # print "Old Tunings" + str(old_tunings)

    return old_tunings


# Form the neighbour using random index and random value selection
def neighbour_selection(parameters, old_tunings):

    dic_length = len(parameters)
    random_index = randint(0,dic_length-1)

    random_index_tuning_value = np.random.uniform(parameters[random_index]['low'], parameters[random_index]['high'])

    if random_index != 0:
        random_index_tuning_value = int(random_index_tuning_value)

    return random_index,random_index_tuning_value


# Acceptance probablility calculation

def acceptance_probablitiy(old_cost, new_cost, temperature):

    power_factor = (old_cost - new_cost)/temperature
    ap = math.exp(power_factor)

    return ap


# Function for simulated annealing

def simulated_annealing_random(dataset):

    # Sets the valid range for each parameter of the machine learning algorithm
    algoParameters = [{'low': 0, 'high': 1}, {'low': 1, 'high': 20}, {'low': 2, 'high': 20}, {'low': 1, 'high': 20}, {'low': 1, 'high': 50}]

    old_tunings = population_initialisation(algoParameters)

    old_cost = score(old_tunings, dataset)

    T = 1.0
    T_min = 0.00001
    alpha = 0.9

    while T >= T_min:

        i = 1

        print "Iteration at Temp" + str(T) + "\n"

        while i <= 100:

            print "Iteration number : " + str(i)

            new_tuning_index, new_tuning_index_value = neighbour_selection(algoParameters, old_tunings)
            new_tunings = old_tunings
            new_tunings[new_tuning_index] = new_tuning_index_value

            #print "Random index selected : " + str(new_tuning_index)
            #print "Random index - Random value selected : " + str(new_tuning_index_value)
            #print "New Tunings" + str(new_tunings)

            #print "\n"

            new_cost = score(new_tunings, dataset)

            ap = acceptance_probablitiy(old_cost, new_cost, T)

            if ap > np.random.uniform(0, 1):

                old_tunings = new_tunings
                old_cost = new_cost

            i += 1

        T = T * alpha

    return old_tunings, old_cost


if __name__ == "__main__":

    for i in range(0,17):

        print "Dataset under consideration : " + str(i) + "\n"

        x = simulated_annealing_random(i)
        test_score = score_test(x[0], i)

        local_data = {}

        local_data["Dataset No : "] = i
        local_data["Test Score"] = test_score

        global_data.append(local_data)

    print "\n" + str(global_data)

