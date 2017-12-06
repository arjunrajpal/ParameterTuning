from random import random,randint
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, f1_score
import pandas as pd
import numpy as nump
import math
from data import data
from prettytable import PrettyTable


def initialiseSolution(no_of_parameters):

    tunings = []

    for i in range(0, no_of_parameters):

        tuning = nump.random.uniform(algoParameters[i]['low'], algoParameters[i]['high'])

        if (i != 0):
            tuning = int(tuning)

        tunings.append(tuning)

    solution = {"tunings": tunings}

    return solution


# Runs CART on the dataset using parameters present in candidate


# Runs Random Forest on the dataset using parameters present in candidate
def cart(a, b, c, d, candidate):
    # print "\nCandidate in Cart : " + str(candidate)
    # # value_threshold = nump.random.uniform(0.01, 1)
    # # print "Threshold :" + str(value_threshold)
    # print "Threshold :" + str(candidate["tunings"][0])
    # print "Max feature : " + str(candidate["tunings"][1])
    # print "Min sample split : " + str(candidate["tunings"][2])
    # print "Min samples leaf : " + str(candidate["tunings"][3])
    # print "Max depth : " + str(candidate["tunings"][4])

    ct = DecisionTreeClassifier(max_depth=candidate["tunings"][4], min_samples_split=candidate["tunings"][2], min_samples_leaf=candidate["tunings"][3], max_features=candidate["tunings"][1], min_impurity_split=candidate["tunings"][0])
    ct.fit(a, b)
    pred = ct.predict(c)
    # print "Predicted Matrix : "  + str(pred)
    p = f1_score(d, pred, average='weighted')

    return p


# Retrieves the appropriate dataset from readDataset function present in data.py and passes
# the Training and Testing dataset to Random Forest function
def score(candidate, dataset):

    df1, df2 = data.readDataset(dataset)

    # Training Set
    X_Train_DF = df1.ix[:, 3:23]
    X_Train = X_Train_DF.values.astype(float)
    Y_Train = nump.asarray(list(df1["bug"]))
    # print X_Train
    # print Y_Train

    # Testing Set
    X_Test_DF = df2.ix[:, 3:23]
    X_Test = X_Test_DF.values.astype(float)
    Y_Test = nump.asarray(list(df2["bug"]))
    # print X_Test
    # print Y_Test

    p = cart(X_Train, Y_Train, X_Test, Y_Test, candidate)

    return p


def score_test(candidate, dataset):

    df1, df2 = data.readDataset_for_testing(dataset)

    # Training Set
    X_Train_DF = df1.ix[:, 3:23]
    X_Train = X_Train_DF.values.astype(float)
    Y_Train = nump.asarray(list(df1["bug"]))
    # print X_Train
    # print Y_Train

    # Testing Set
    X_Test_DF = df2.ix[:, 3:23]
    X_Test = X_Test_DF.values.astype(float)
    Y_Test = nump.asarray(list(df2["bug"]))
    # print X_Test
    # print Y_Test

    p = cart(X_Train, Y_Train, X_Test, Y_Test, candidate)

    return p


def acceptance_probability(cost,new_cost,T):
    delta_cost = new_cost-cost
    ap = nump.exp(delta_cost/T)

    return ap


def neighbour(tunings,no_of_parameters):
    i = randint(0,no_of_parameters-1)
    tunings[i] = tunings[i] + nump.random.uniform(0,1)
    tunings[i] = max(algoParameters[i]['low'], min(tunings[i], algoParameters[i]['high']))

    if i != 0:
        tunings[i] = int(tunings[i])

    new_solution = {"tunings":tunings}

    return new_solution


# k is iteration no in outer loop
def update_temp_fast_schedule(k,T,n=1.0,quench=1.0):

    c = n * nump.exp(-n*quench)
    T_new = T*nump.exp(-c*k**quench)
    return T_new


def fast_schedule(tunings,no_of_parameters,T):
    # print T
    u = nump.random.uniform(0,1,size=nump.asarray(tunings).shape)
    y = nump.sign(u-0.5)*T*((1+(1/T))**abs(2*u-1) - 1.0)
    # print u
    # print "U:"+str(u)+"\n Y:"+str(y)+" sk "+str((1+(1/T)))

    for i in range(len(tunings)):
        xc = y[i] * (algoParameters[i]['high']-algoParameters[i]['low'])
        tunings[i] = tunings[i] + xc
        if(i!=0):
            tunings[i] = int(tunings[i])

        tunings[i] = max(algoParameters[i]['low'], min(tunings[i], algoParameters[i]['high']))

    new_solution = {"tunings":tunings}

    return new_solution


def update_temp_cauchy_schedule(k,T):
    return T/(1+k)


def cauchy_schedule(tunings,T,learning_rate=0.5):
    u = nump.random.uniform(-(math.pi/2),math.pi/2,size=nump.asarray(tunings).shape)

    # xc = learning_rate*T*nump.tan(u)
    # tunings = tunings + xc

    # for i in range(len(tunings)):
    #     if i!=0:
    #         tunings[i] = int(tunings[i])
    #     tunings[i] = max(algoParameters[i]['low'], min(tunings[i], algoParameters[i]['high']))

    # Alternate method
    for i in range(len(tunings)):
        xc = learning_rate*T*math.tan(u[i])
        tunings[i] = tunings[i] + xc
        if i!=0:
            tunings[i] = int(tunings[i])
        tunings[i] = max(algoParameters[i]['low'], min(tunings[i], algoParameters[i]['high']))

    new_solution = {"tunings":tunings}

    return new_solution


def SA(no_of_iterations, T, T_min, alpha, no_of_parameters, dataset):

    T_initial = T
    solution = initialiseSolution(no_of_parameters)
    solution['cost'] = score(solution,dataset)

    count=0

    while T>T_min:

        i=1
        while i<= no_of_iterations:
            new_solution = fast_schedule(solution['tunings'],no_of_parameters,T)

            new_solution['cost'] = score(new_solution,dataset)

            if new_solution['cost']  > solution['cost']:
                solution = new_solution
            else:
                ap = acceptance_probability(solution['cost'],new_solution['cost'],T)

                if ap>nump.random.uniform(0,1):
                    solution = new_solution

            i = i+1

        # print "Iteration at Temp" + str(T)
        T = update_temp_fast_schedule(count,T_initial)
        count += 1

    return solution

# Sets the valid range for each parameter of the machine learning algorithm


algoParameters = [{'low': 0, 'high': 1}, {'low': 1, 'high': 5}, {'low': 2, 'high': 20}, {'low': 1, 'high': 20}, {'low': 1, 'high': 50}]

# Invokes SA

all_data_fscore_CART = []

print "Fscore CART using Fast schedule"


def calculate():
    for i in range(0, 17):
        dataset = i
        parameters = SA(100, 1.0, 0.00001, 0.9, 5, dataset)
        score_p = (score_test(parameters, dataset) * 100)
        print "Dataset :" + str(i) + " Parameters :" + str(parameters) + " Score :" + str(score_p)
        all_data_fscore_CART.append(score_p)

    print all_data_fscore_CART
    return all_data_fscore_CART

if __name__ == "__main__":
    calculate()
