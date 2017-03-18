import numpy as nump
import math

np, f, cr, life, Goal, noOfParameters = 10, 0.75, 0.3, 5, [] #input
sBest = [] #ouput

# def Score(candidate):
#
#

def initialisePopulation(np,noOfParameters):
    population = [{'tunings':[],'score':0}]

    # population = nump.zeros((np,noOfParameters))
    # population = nump.reshape(population,(np,noOfParameters))

    for i in range(0,np):
        population[i][0] = nump.random.uniform(0.01,1.01)
        population[i][1] = nump.random.uniform(0.01,1.01)
        population[i][2] = nump.random.uniform(1,50.01)
        population[i][3] = nump.random.uniform(2,20.01)
        population[i][4] = nump.random.uniform(1,20.01)
        population[i][5] = nump.random.uniform(50,150.01)

    print population

def threeOthers(pop,old):
    three = nump.random.randint(0, 10, size=(1, 3))

    return three[0], three[1], three[2]


def improve(population, oldGeneration):
    if population != oldGeneration:
        return True
    else:
        return False

def getBestSolution(population):
    max = -1
    bestSolution = []

    for i in range(0,population.size):
        scores = score(population[i])
        if scores > max:
            max = scores
            bestSolution = population[i]

    return bestSolution


def extrapolate(old, pop, cr, f):
    a, b, c = threeOthers(pop,old)
    newf = []

    for i in range(0,noOfParameters):
        if cr<random(0,1):
            newf.append(old[i])
        elif type(old[i]) == bool:
                newf.append(not old[i])
        else:
            newf.append(a[i]+f*(b[i]-c[i]))


    return newf

def DE(np, f, cr, life, Goal):
    population = initialisePopulation(np,noOfParameters)
    sBest = getBestSolution(population)

    while life > 0:
        newGeneration = nump.zeros((10,noOfParameters))

        for i in range(0,np):
            s[i] = extrapolate(population[i],population,cr,f)
            if score(s[i])>score(population[i]):
                newGeneration.append(s[i])
            else:
                newGeneration.append(population[i])

        oldGeneration = population
        population = newGeneration

        if not improve(population, oldGeneration):
            life -= 1

        sBest = getBestSolution(population)

    return sBest

# initialisePopulation(10,6)


