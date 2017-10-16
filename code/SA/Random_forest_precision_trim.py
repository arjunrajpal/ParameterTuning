from random import random,randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import numpy as np
import math
from data import data


def initialiseSolution(no_of_parameters):

	tunings = []
	for i in range(0,no_of_parameters):
		tuning = np.random.uniform(algoParameters[i]['low'],algoParameters[i]['high'])

		if(i!=0):
			tuning = int(tuning)

		tunings.append(tuning)

	solution = {"tunings":tunings}

	return solution

# Runs Random Forest on the dataset using parameters present in candidate
def random_forest(a, b, c, d, candidate):
    # print "Candidate in Random Forest : " + str(candidate)
    # print "Threshold :" + str(candidate["tunings"][0])
    # print "Max feature : " + str(candidate["tunings"][1])
    # print "Max leaf nodes : " + str(candidate["tunings"][2])
    # print "Min sample split : " + str(candidate["tunings"][3])
    # print "Min samples leaf : " + str(candidate["tunings"][4])
    # print "Max no of estimators : " + str(candidate["tunings"][5])

    rf = RandomForestClassifier(min_impurity_split=candidate["tunings"][0], max_features=candidate['tunings'][1],
                                max_leaf_nodes=candidate['tunings'][2], min_samples_split=candidate['tunings'][3],
                                min_samples_leaf=candidate['tunings'][4], n_estimators=candidate['tunings'][5])
    rf.fit(a, b)
    pred = rf.predict(c)
    # print "Predicted Matrix : "  + str(pred)
    p = precision_score(d, pred, average='weighted')

    return p

# Retrieves the appropriate dataset from readDataset function present in data.py and passes
# the Training and Testing dataset to Random Forest function
# Retrieves the appropriate dataset from readDataset function present in data.py and passes
# the Training and Testing dataset to Random Forest function
def score(candidate, datasets):

    df1, df2 = data.readDataset(datasets)

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

    p = random_forest(X_Train, Y_Train, X_Test, Y_Test, candidate)

    return p


def score_test(candidate, datasets):

    df1, df2 = data.readDataset_for_testing(datasets)

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

    p = random_forest(X_Train, Y_Train, X_Test, Y_Test, candidate)

    return p

def acceptance_probability(new_cost,cost,T):
	delta_cost = cost-new_cost
	ap = np.exp(delta_cost/T)

	return ap 


def neighbour(tunings,no_of_parameters):

	i = randint(0,no_of_parameters-1)
	tunings[i] = tunings[i] + np.random.uniform(0,1)
	tunings[i] = max(algoParameters[i]['low'], min(tunings[i], algoParameters[i]['high']))
	if i != 0:
			tunings[i] = int(tunings[i])
	new_solution = {"tunings":tunings}
   	return new_solution


def SA(no_of_iterations,T,T_min,alpha,no_of_parameters,dataset):
	solution = initialiseSolution(no_of_parameters)
	solution['cost'] = score(solution,dataset)

	while T>T_min:

		i=1

		while i<= no_of_iterations:

			new_solution = neighbour(solution['tunings'],no_of_parameters)
			new_solution['cost'] = score(new_solution,dataset)

			ap = acceptance_probability(solution['cost'],new_solution['cost'],T)

			if ap>np.random.uniform(0, 1):

				solution = new_solution

			i=i+1
		
		print "Iteration at Temp" + str(T)

		T = T*alpha

	return solution

# Sets the valid range for each parameter of the machine learning algorithm
algoParameters = [{'low': 0.01, 'high': 1}, {'low': 1, 'high': 20}, {'low': 2, 'high': 50}, {'low': 2, 'high': 20},
                  {'low': 1, 'high': 20}, {'low': 50, 'high': 150}]

# Invokes SA

all_data_precision_rf = []

print "Precision Random Forest"

def calculate():
    for i in range(2, 3):
        dataset = i
        parameters = SA(50, 1.0, 0.01, 0.9, 6, dataset)
        score_p = (score_test(parameters, dataset) * 100)
        print "Dataset :" + str(i) + " Parameters :" + str(parameters) + " Score :" + str(score_p)
        all_data_precision_rf.append(score_p)

    print all_data_precision_rf

if __name__ == "__main__":
    calculate()
