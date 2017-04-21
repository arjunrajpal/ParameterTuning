from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
import pandas as pd
import numpy as nump
import math
from data import data
import random

# np, f, cr, life, Goal, noOfParameters = 10, 0.75, 0.3, 5, [], 0 #input
# sBest = [] #ouput

global global_best_score
global_best_score = []
global global_best
global_best= []


# Runs Random Forest on the dataset using parameters present in candidate
def random_forest(a, b, c, d, candidate):
    print "Candidate in Random Forest : " + str(candidate)
    # value_threshold = nump.random.uniform(0.01, 1)
    # print "Threshold :" + str(value_threshold)
    print "Threshold :" + str(candidate["tunings"][0])
    print "Max feature : " + str(candidate["tunings"][1])
    print "Max leaf nodes : " + str(candidate["tunings"][2])
    print "Min sample split : " + str(candidate["tunings"][3])
    print "Min samples leaf : " + str(candidate["tunings"][4])
    print "Max no of estimators : " + str(candidate["tunings"][5])

    rf = RandomForestClassifier(n_estimators=candidate['tunings'][5], min_samples_split=candidate['tunings'][3], min_samples_leaf=candidate['tunings'][4], max_features = candidate['tunings'][1], max_leaf_nodes = candidate['tunings'][2], min_impurity_split=candidate["tunings"][0])
    rf.fit(a, b)
    pred = rf.predict(c)
    # print "Predicted Matrix : "  + str(pred)
    p = f1_score(d, pred, average='weighted')

    return p


# Retrieves the appropriate dataset from readDataset function present in data.py and passes
# the Training and Testing dataset to Random Forest function
def score(candidate, datasets):

    df1, df2 = data.readDataset(datasets)

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

    p = random_forest(X_Train, Y_Train, X_Test, Y_Test, candidate)

    return p


def score_test(candidate, datasets):

    df1, df2 = data.readDataset_for_testing(datasets)

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

    p = random_forest(X_Train, Y_Train, X_Test, Y_Test, candidate)

    return p


# Initialises the population such that each parameter of each candidate
# in the population gets a random value from the parameter's valid range
def initialisePopulation(np,noOfParameters):
    population = []

    for i in range(0,np):
        candidate = {}
        tunings = []
        # tunings.append(nump.random.uniform(algoParameters[0]['low'], algoParameters[0]['high']))

        threshold= nump.random.uniform(algoParameters[0]['low'], algoParameters[0]['high'])
        print "Threshold for index " + str(i) + " : " + str(threshold)
        tunings.append(threshold)

        max_feature = int(nump.random.uniform(algoParameters[1]['low'], algoParameters[1]['high']))
        print "Max Feature selected for index " + str(i) + " : " + str(max_feature)
        tunings.append(max_feature)

        max_leaf_nodes = int(nump.random.uniform(algoParameters[2]['low'], algoParameters[2]['high']))
        print "Max Leaf Nodes for index " + str(i) + " : " + str(max_leaf_nodes)
        tunings.append(max_leaf_nodes)

        min_sample_split = int(nump.random.uniform(algoParameters[3]['low'], algoParameters[3]['high']))
        print "Min sample split for index " + str(i) + " : " + str(min_sample_split)
        tunings.append(min_sample_split)

        min_samples_leaf = int(nump.random.uniform(algoParameters[4]['low'], algoParameters[4]['high']))
        print "Min samples leaf for index " + str(i) + " : " + str(min_samples_leaf)
        tunings.append(min_samples_leaf)

        n_estimators = int(nump.random.uniform(algoParameters[5]['low'], algoParameters[5]['high']))
        print "n_estimators for index " + str(i) + " : " + str(n_estimators) + "\n"
        tunings.append(n_estimators)

        # population[i]['tunings']
        # print tunings[4]
        candidate['tunings'] = tunings
        candidate['score'] = 0

        population.append(candidate)

    print "Population"
    print population

    return population


# Returns three random candidates from a population of size 10 such that
# neither of them is equal to the target candidate i.e old
def threeOthers(pop,old,index):

    # while(1):
    three = range(0,10,1)
    three.remove(index)
    three = random.sample(three, 3)  # Array Formation

    # print pop[three[0]['tunings']]
    # print pop[three[0]['tunings']]
    # print pop[three[0]['tunings']]

    return pop[three[0]]['tunings'], pop[three[1]]['tunings'], pop[three[2]]['tunings']


# Returns True if the newly computed population is different than the old generation otherwise False
def improve(population, oldGeneration):
    if population != oldGeneration:
        return True
    else:
        return False


# Compares the f1_score of each candidate in the population and
# returns the candidate with the highest f1_score as the Best Solution
def getBestSolution(population):
    max = -1
    bestSolution = {}

    for i in range(0,len(population)):
        scores = population[i]['score']
        # print scores

        if scores > max:
            max = scores
            bestSolution = population[i]

    return bestSolution


# Creates a mutant whose each parameter value is either taken from the target candidate(old) or is
# computed using (a+f*(b-c)) depending on value of cr
def extrapolate(old, pop, cr, f, noOfParameters, index):
    a, b, c = threeOthers(pop, old, index)  # index is for the target row

    newf = []

    print "\nThe other 3 selected rows for index " + str(index) + " : "
    print a
    print b
    print c

    for i in range(0, noOfParameters):

        x = nump.random.uniform(0, 1)
        print "Random number for comparison with cr : " + str(x)

        if cr < x:
            print "Old tuning Value for index " + str(index) + " : " + (str(old['tunings'][i]))
            newf.append(old['tunings'][i])

        elif type(old['tunings'][i]) == bool:
                newf.append(not old['tunings'][i])
        else:
            lo = algoParameters[i]["low"]
            hi = algoParameters[i]["high"]
            value = a[i] + (f * (b[i] - c[i]))

            print "Value before trim : " + str(value)

            if i != 0:
                mutant_value = int(max(lo, min(value, hi)))
            else:
                mutant_value = max(lo, min(value, hi))

            print "Mutant Value : " + str(mutant_value)

            newf.append(mutant_value)

    dict_mutant = {'tunings': newf}
    score_mutant = score(dict_mutant, datasets)

    score_original = score(old, datasets)

    print "Original Score : " + str(score_original)
    print "Mutant Score : " + str(score_mutant)

    global global_best
    global global_best_score

    if score_mutant > score_original:
        global_best_score.append(score_mutant)
        global_best.append({'score': score_mutant, 'tunings': newf})

    else:
        global_best_score.append(score_original)
        global_best.append({'score': score_original, 'tunings': old["tunings"]})

    print global_best

    # newCandidate = {'score': 0, 'tunings': newf}
    # print newCandidate
    # return newCandidate


# Performs Differential Evolution
def DE(np, f, cr, life, noOfParameters, datasets):

    population = initialisePopulation(np, noOfParameters)  # Intial population formation

    while life > 0:

        global global_best
        global_best = []
        global global_best_score
        global_best_score = []

        for i in range(0, np):
            extrapolate(population[i], population, cr, f, noOfParameters, i)

        print "Global Best :"
        print global_best

        oldPopulation = []
        globalPopulation = []

        for row in population:
            oldPopulation.append(row['tunings'])

        for row in global_best:
            globalPopulation.append(row['tunings'])

        print "Old Population :"
        print oldPopulation

        print "Global Population :"
        print globalPopulation

        if oldPopulation != globalPopulation:
            population = global_best
            print population
        else:
          life -= 1

        s_Best = getBestSolution(global_best)

    return s_Best

# Sets the valid range for each parameter of the machine learning algorithm
algoParameters = [{'low': 0.01, 'high': 1},{'low': 1, 'high': 17}, {'low': 2, 'high': 50}, {'low': 2, 'high': 20}, {'low': 1, 'high': 20}, {'low': 50, 'high': 150}]

# Invokes DE

datasets = raw_input("Enter dataset : ")
datasets = int(datasets)

parameters = DE(10, 0.75, 0.3, 5, 6, datasets)

print "\nBest Parameters for Random Forest in dataset ",datasets," are ", parameters
print score_test(parameters, datasets)
