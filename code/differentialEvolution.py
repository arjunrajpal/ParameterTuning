from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import pandas as pd
import numpy as nump
import math

np, f, cr, life, Goal, noOfParameters = 10, 0.75, 0.3, 5, [], 0 #input
sBest = [] #ouput

def random_forest(a,b,c,d, candidate):
    # print candidate
    rf = RandomForestClassifier(max_features=candidate['tunings'][0], max_leaf_nodes=candidate['tunings'][1], min_samples_split=candidate['tunings'][2], min_samples_leaf=candidate['tunings'][3], n_estimators=candidate['tunings'][4])
    rf.fit(a,b)
    pred = rf.predict(c)
    # print "Predicted Matrix : "  + str(pred)
    p = precision_score(d, pred, average='macro')

    return p

def score(candidate):
    # antV1

    f1 = pd.read_csv("../final_dataset/ant/2.csv", delimiter=",")
    df1 = pd.DataFrame(f1)

    f2 = pd.read_csv("../final_dataset/ant/4.csv", delimiter=",")
    df2 = pd.DataFrame(f2)

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


def initialisePopulation(np,noOfParameters):
    population = []

    for i in range(0,np):
        candidate = {}
        tunings = []
        # tunings.append(nump.random.uniform(0.01,1))
        tunings.append(nump.random.uniform(0.01,1))
        tunings.append(nump.random.randint(1,50))
        tunings.append(nump.random.randint(2,20))
        tunings.append(nump.random.randint(1,20))
        tunings.append(nump.random.randint(50,150))

        # population[i]['tunings']
        # print tunings[4]
        candidate['tunings'] = tunings
        candidate['score'] = 0

        population.append(candidate)

    # print population

    return population

def threeOthers(pop,old):

    while(1):
        three = nump.random.randint(0, 10, size=(3))
        three = list(three)

        # print three

        if three[0] != three[1] and three[1] != three[2] and three[0] != three[2]:
            if (pop[three[0]] != old) and (pop[three[1]] != old) and (pop[three[2]] != old):
                break

    return pop[three[0]]['tunings'], pop[three[1]]['tunings'], pop[three[2]]['tunings']


def improve(population, oldGeneration):
    if population != oldGeneration:
        return True
    else:
        return False

def getBestSolution(population):
    max = -1

    for i in range(0,len(population)):
        scores = population[i]['score']
        if scores > max:
            max = scores
            bestSolution = population[i]

    return bestSolution


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
            newf.append(a[i]+f*(b[i]-c[i]))

    newf[1] = int(newf[1])
    newf[2] = int(newf[2])
    newf[3] = int(newf[3])
    newf[4] = int(newf[4])

    if(newf[0] < 0.01) or (newf[0] > 1):
        newf[0] = nump.random.uniform(0.01,1)

    if(newf[1] < 1) or (newf[1] > 50):
        newf[1] = nump.random.randint(1,50)

    if (newf[2] < 2) or (newf[2] > 20):
        newf[2] = nump.random.randint(2,20)

    if (newf[3] < 1) or (newf[3] > 20):
        newf[3] = nump.random.randint(1,20)

    if (newf[4] < 50) or (newf[4] > 150):
        newf[4] = nump.random.randint(50,150)

    newCandidate = {'score':0, 'tunings':newf}

    # print newCandidate

    return newCandidate

def DE(np, f, cr, life, noOfParameters):
    population = initialisePopulation(np,noOfParameters)
    sBest = getBestSolution(population)

    while life > 0:
        newGeneration = []

        for i in range(0,np):
            s = extrapolate(population[i],population,cr,f,noOfParameters)

            score_s = score(s)
            score_population_i = score(population[i])

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

    return sBest

# initialisePopulation(10,6)

print "Best Parameters for Random Forest", DE(10, 0.75, 0.3, 5, 5)

