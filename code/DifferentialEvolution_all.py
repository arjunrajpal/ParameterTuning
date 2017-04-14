from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score
import pandas as pd
import numpy as nump
import math
from data import data

# np, f, cr, life, Goal, noOfParameters = 10, 0.75, 0.3, 5, [], 0 #input
# sBest = [] #ouput


# Runs Random Forest on the dataset using parameters present in candidate
def random_forest(a,b,c,d,candidate):
    # print candidate
    rf = RandomForestClassifier(max_features=candidate['tunings'][0], max_leaf_nodes=candidate['tunings'][1], min_samples_split=candidate['tunings'][2], min_samples_leaf=candidate['tunings'][3], n_estimators=candidate['tunings'][4])
    rf.fit(a,b)
    pred = rf.predict(c)
    # print "Predicted Matrix : "  + str(pred)
    p = precision_score(d, pred, average='macro')

    return p


# Runs CART on the dataset
# def cart(a,b,c,d):
#     ct = DecisionTreeClassifier()
#     ct.classes_ = nump.arange(1,29)
#     ct.fit(a,b)
#     pred = ct.predict(c)
#     # print "Predicted Matrix : "  + str(pred)
#     p = precision_score(d, pred, average='macro')
#
#     return p

# Determines which algo to run depending on the value of mlago
# def runAlgo(X_Train, Y_Train, X_Test, Y_Test, candidate, mlalgo):
#
#     if mlalgo == 0:
#         p = random_forest(X_Train, Y_Train, X_Test, Y_Test, candidate)
#     elif mlalgo == 1:
#         p = cart(X_Train, Y_Train, X_Test, Y_Test, candidate)
#
#     return p


# Limits the newly computed value to the legal range min...max of that parameter (decision)
def trim(decision, computed):
    return max(decision['high'], min(computed, decision['low']))


# Retrieves the appropriate dataset from readDataset function present in data.py and passes the Training and Testing dataset to Random Forest function
def score(candidate, datasets):

    # # antV1
    #
    # f1 = pd.read_csv("../final_dataset/ant/2.csv", delimiter=",")
    # df1 = pd.DataFrame(f1)
    #
    # f2 = pd.read_csv("../final_dataset/ant/4.csv", delimiter=",")
    # df2 = pd.DataFrame(f2)

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

    # p = random_forest(X_Train, Y_Train, X_Test, Y_Test, candidate)

    p = random_forest(X_Train, Y_Train, X_Test, Y_Test, candidate)

    return p

# Initialises the population such that each parameter of each candidate in the population gets a random value from the parameter's valid range
def initialisePopulation(np,noOfParameters):
    population = []

    for i in range(0,np):
        candidate = {}
        tunings = []
        # tunings.append(nump.random.uniform(0.01,1))
        tunings.append(nump.random.uniform(algoParameters[1]['low'],algoParameters[1]['high']))
        tunings.append(nump.random.randint(algoParameters[2]['low'],algoParameters[2]['high']))
        tunings.append(nump.random.randint(algoParameters[3]['low'],algoParameters[3]['high']))
        tunings.append(nump.random.randint(algoParameters[4]['low'],algoParameters[4]['high']))
        tunings.append(nump.random.randint(algoParameters[5]['low'],algoParameters[5]['high']))

        # population[i]['tunings']
        # print tunings[4]
        candidate['tunings'] = tunings
        candidate['score'] = 0

        population.append(candidate)

    # print population

    return population


# Returns three random candidates from a population of size 10 such that neither of them is equal to the target candidate i.e old
def threeOthers(pop,old):

    while(1):
        three = nump.random.randint(0, 10, size=(3))
        three = list(three)

        # print three

        if three[0] != three[1] and three[1] != three[2] and three[0] != three[2]:
            if (pop[three[0]] != old) and (pop[three[1]] != old) and (pop[three[2]] != old):
                break

    return pop[three[0]]['tunings'], pop[three[1]]['tunings'], pop[three[2]]['tunings']


# Returns True if the newly computed population is different than the old generation otherwise False
def improve(population, oldGeneration):
    if population != oldGeneration:
        return True
    else:
        return False


# Compares the precision score of each candidate in the population and returns the candidate with the highest precision score as the Best Solution
def getBestSolution(population):
    max = -1
    bestSolution = {}

    for i in range(0,len(population)):
        scores = population[i]['score']
        if scores > max:
            max = scores
            bestSolution = population[i]

    return bestSolution


# Creates a mutant whose each parameter value is either taken from the target candidate(old) or is computed using (a+f*(b-c)) depending on value of cr
def extrapolate(old, pop, cr, f, noOfParameters):
    a, b, c = threeOthers(pop,old)
    newf = []

    # print a,b,c
    for i in range(0,noOfParameters):
        if cr < nump.random.uniform(0,1):
            newf.append(old['tunings'][i])
        elif type(old['tunings'][i]) == bool:
                newf.append(not old['tunings'][i])
        else:
            newf.append(trim(algoParameters[i+1],a[i]+f*(b[i]-c[i])))

    # newf[1] = int(newf[1])
    # newf[2] = int(newf[2])
    # newf[3] = int(newf[3])
    # newf[4] = int(newf[4])

    # if(newf[0] < 0.01) or (newf[0] > 1):
    #     newf[0] = math.m(0.01,1)
    #
    # if(newf[1] < 1) or (newf[1] > 50):
    #     newf[1] = nump.random.randint(1,50)
    #
    # if (newf[2] < 2) or (newf[2] > 20):
    #     newf[2] = nump.random.randint(2,20)
    #
    # if (newf[3] < 1) or (newf[3] > 20):
    #     newf[3] = nump.random.randint(1,20)
    #
    # if (newf[4] < 50) or (newf[4] > 150):
    #     newf[4] = nump.random.randint(50,150)

    newCandidate = {'score':0, 'tunings':newf}

    # print newCandidate

    return newCandidate


# Performs Differential Evolution
def DE(np, f, cr, life, noOfParameters, datasets):

    sBest = []
    population = initialisePopulation(np,noOfParameters)
    sBest = getBestSolution(population)

    while life > 0:
        newGeneration = []

        for i in range(0,np):
            s = extrapolate(population[i],population,cr,f,noOfParameters)

            score_s = score(s,datasets)
            score_population_i = score(population[i],datasets)

            s['score'] = score_s
            population[i]['score'] = score_population_i

            if score_s > score_population_i:
                newGeneration.append(s)
            else:
                newGeneration.append(population[i])

        oldGeneration = population
        population = newGeneration

        if not improve(population, oldGeneration):
            life -= 1

        sBest = getBestSolution(population)
        # print sBest
    return sBest


# initialisePopulation(10,6)
# print "Best Parameters for Random Forest", DE(10, 0.75, 0.3, 5, 5)

# Sets the valid range for each parameter of the machine learning algorithm
algoParameters = [{'low':0.01, 'high':1}, {'low':0.01, 'high':1}, {'low':1, 'high':50}, {'low':2, 'high':20}, {'low':1, 'high':20}, {'low':50, 'high':150}]

# Invokes DE
for datasets in range(17):
    parameters = DE(10, 0.75, 0.3, 5, 5, datasets)

    print "Best Parameters for Random Forest are ", parameters




