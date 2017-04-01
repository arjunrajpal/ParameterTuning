from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score,accuracy_score, f1_score
import matplotlib.pyplot as plt

precision_lr =[]
precision_rf = []
precision_ct = []

def logistic_regression(a, b, c,d):
    lr = LogisticRegression()
    lr.fit(a,b)
    pred = lr.predict(c)
    # print "Predicted Matrix : "  + str(pred)
    p = precision_score(d, pred, average='macro')
    precision_lr.append(p * 100)


def random_forest(a,b,c,d):
    rf = RandomForestClassifier()
    rf.fit(a,b)
    pred = rf.predict(c)
    # print "Predicted Matrix : "  + str(pred)
    p = precision_score(d, pred, average='macro')
    f = f1_score(d, pred, average='macro')
    # print f
    precision_rf.append(p * 100)

def cart(a,b,c,d):
    ct = DecisionTreeClassifier()
    ct.classes_ = np.arange(1,29)
    ct.fit(a,b)
    pred = ct.predict(c)
    # print "Predicted Matrix : "  + str(pred)
    p = precision_score(d, pred, average='macro')
    precision_ct.append(p * 100)

# antV0
f1 = pd.read_csv("../final_dataset_untuned/ant/antV0/1.csv", delimiter = ",")
df1 = pd.DataFrame(f1)

f2 = pd.read_csv("../final_dataset_untuned/ant/antV0/3.csv", delimiter = ",")
df2 = pd.DataFrame(f2)

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

logistic_regression(X_Train, Y_Train, X_Test, Y_Test)
random_forest(X_Train, Y_Train, X_Test, Y_Test)
cart(X_Train, Y_Train, X_Test, Y_Test)

# antV1

f1 = pd.read_csv("../final_dataset_untuned/ant/antV1/2.csv", delimiter = ",")
df1 = pd.DataFrame(f1)

f2 = pd.read_csv("../final_dataset_untuned/ant/antV1/4.csv", delimiter = ",")
df2 = pd.DataFrame(f2)

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

logistic_regression(X_Train, Y_Train, X_Test, Y_Test)
random_forest(X_Train, Y_Train, X_Test, Y_Test)
cart(X_Train, Y_Train, X_Test, Y_Test)


# antV2

f1 = pd.read_csv("../final_dataset_untuned/ant/antV2/3.csv", delimiter = ",")
df1 = pd.DataFrame(f1)

f2 = pd.read_csv("../final_dataset_untuned/ant/antV2/5.csv", delimiter = ",")
df2 = pd.DataFrame(f2)

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

logistic_regression(X_Train, Y_Train, X_Test, Y_Test)
random_forest(X_Train, Y_Train, X_Test, Y_Test)
cart(X_Train, Y_Train, X_Test, Y_Test)


# camelV0

f1 = pd.read_csv("../final_dataset_untuned/camel/camelV0/1.csv", delimiter = ",")
df1 = pd.DataFrame(f1)

f2 = pd.read_csv("../final_dataset_untuned/camel/camelV0/3.csv", delimiter = ",")
df2 = pd.DataFrame(f2)

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

logistic_regression(X_Train, Y_Train, X_Test, Y_Test)
random_forest(X_Train, Y_Train, X_Test, Y_Test)
cart(X_Train, Y_Train, X_Test, Y_Test)


# camelV1

f1 = pd.read_csv("../final_dataset_untuned/camel/camelV1/2.csv", delimiter = ",")
df1 = pd.DataFrame(f1)

f2 = pd.read_csv("../final_dataset_untuned/camel/camelV1/4.csv", delimiter = ",")
df2 = pd.DataFrame(f2)

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

logistic_regression(X_Train, Y_Train, X_Test, Y_Test)
random_forest(X_Train, Y_Train, X_Test, Y_Test)
cart(X_Train, Y_Train, X_Test, Y_Test)


# ivy

f1 = pd.read_csv("../final_dataset_untuned/ivy/1.csv", delimiter = ",")
df1 = pd.DataFrame(f1)

f2 = pd.read_csv("../final_dataset_untuned/ivy/3.csv", delimiter = ",")
df2 = pd.DataFrame(f2)

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

logistic_regression(X_Train, Y_Train, X_Test, Y_Test)
random_forest(X_Train, Y_Train, X_Test, Y_Test)
cart(X_Train, Y_Train, X_Test, Y_Test)


# jeditV0

f1 = pd.read_csv("../final_dataset_untuned/jedit/jeditV0/1.csv", delimiter = ",")
df1 = pd.DataFrame(f1)

f2 = pd.read_csv("../final_dataset_untuned/jedit/jeditV0/3.csv", delimiter = ",")
df2 = pd.DataFrame(f2)

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

logistic_regression(X_Train, Y_Train, X_Test, Y_Test)
random_forest(X_Train, Y_Train, X_Test, Y_Test)
cart(X_Train, Y_Train, X_Test, Y_Test)


# jeditV1

f1 = pd.read_csv("../final_dataset_untuned/jedit/jeditV1/2.csv", delimiter = ",")
df1 = pd.DataFrame(f1)

f2 = pd.read_csv("../final_dataset_untuned/jedit/jeditV1/4.csv", delimiter = ",")
df2 = pd.DataFrame(f2)

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

logistic_regression(X_Train, Y_Train, X_Test, Y_Test)
random_forest(X_Train, Y_Train, X_Test, Y_Test)
cart(X_Train, Y_Train, X_Test, Y_Test)


# jeditV2

f1 = pd.read_csv("../final_dataset_untuned/jedit/jeditV2/3.csv", delimiter = ",")
df1 = pd.DataFrame(f1)

f2 = pd.read_csv("../final_dataset_untuned/jedit/jeditV2/5.csv", delimiter = ",")
df2 = pd.DataFrame(f2)

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

logistic_regression(X_Train, Y_Train, X_Test, Y_Test)
random_forest(X_Train, Y_Train, X_Test, Y_Test)
cart(X_Train, Y_Train, X_Test, Y_Test)


# log4j

f1 = pd.read_csv("../final_dataset_untuned/log4j/1.csv", delimiter = ",")
df1 = pd.DataFrame(f1)

f2 = pd.read_csv("../final_dataset_untuned/log4j/3.csv", delimiter = ",")
df2 = pd.DataFrame(f2)

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

logistic_regression(X_Train, Y_Train, X_Test, Y_Test)
random_forest(X_Train, Y_Train, X_Test, Y_Test)
cart(X_Train, Y_Train, X_Test, Y_Test)


# lucene

f1 = pd.read_csv("../final_dataset_untuned/lucene/1.csv", delimiter = ",")
df1 = pd.DataFrame(f1)

f2 = pd.read_csv("../final_dataset_untuned/lucene/3.csv", delimiter = ",")
df2 = pd.DataFrame(f2)

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

logistic_regression(X_Train, Y_Train, X_Test, Y_Test)
random_forest(X_Train, Y_Train, X_Test, Y_Test)
cart(X_Train, Y_Train, X_Test, Y_Test)


# poiV0

f1 = pd.read_csv("../final_dataset_untuned/poi/poiV0/1.csv", delimiter = ",")
df1 = pd.DataFrame(f1)

f2 = pd.read_csv("../final_dataset_untuned/poi/poiV0/3.csv", delimiter = ",")
df2 = pd.DataFrame(f2)

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

logistic_regression(X_Train, Y_Train, X_Test, Y_Test)
random_forest(X_Train, Y_Train, X_Test, Y_Test)
cart(X_Train, Y_Train, X_Test, Y_Test)


# poiV1

f1 = pd.read_csv("../final_dataset_untuned/poi/poiV1/2.csv", delimiter = ",")
df1 = pd.DataFrame(f1)

f2 = pd.read_csv("../final_dataset_untuned/poi/poiV1/4.csv", delimiter = ",")
df2 = pd.DataFrame(f2)

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

logistic_regression(X_Train, Y_Train, X_Test, Y_Test)
random_forest(X_Train, Y_Train, X_Test, Y_Test)
cart(X_Train, Y_Train, X_Test, Y_Test)


# synapse

f1 = pd.read_csv("../final_dataset_untuned/synapse/1.csv", delimiter = ",")
df1 = pd.DataFrame(f1)

f2 = pd.read_csv("../final_dataset_untuned/synapse/3.csv", delimiter = ",")
df2 = pd.DataFrame(f2)

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

logistic_regression(X_Train, Y_Train, X_Test, Y_Test)
random_forest(X_Train, Y_Train, X_Test, Y_Test)
cart(X_Train, Y_Train, X_Test, Y_Test)


# velocity

f1 = pd.read_csv("../final_dataset_untuned/velocity/1.csv", delimiter = ",")
df1 = pd.DataFrame(f1)

f2 = pd.read_csv("../final_dataset_untuned/velocity/3.csv", delimiter = ",")
df2 = pd.DataFrame(f2)

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

logistic_regression(X_Train, Y_Train, X_Test, Y_Test)
random_forest(X_Train, Y_Train, X_Test, Y_Test)
cart(X_Train, Y_Train, X_Test, Y_Test)


# xercesV0

f1 = pd.read_csv("../final_dataset_untuned/xerces/xercesV0/1.csv", delimiter = ",")
df1 = pd.DataFrame(f1)

f2 = pd.read_csv("../final_dataset_untuned/xerces/xercesV0/3.csv", delimiter = ",")
df2 = pd.DataFrame(f2)

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

logistic_regression(X_Train, Y_Train, X_Test, Y_Test)
random_forest(X_Train, Y_Train, X_Test, Y_Test)
cart(X_Train, Y_Train, X_Test, Y_Test)


# xercesV1

f1 = pd.read_csv("../final_dataset_untuned/xerces/xercesV1/2.csv", delimiter = ",")
df1 = pd.DataFrame(f1)

f2 = pd.read_csv("../final_dataset_untuned/xerces/xercesV1/4.csv", delimiter = ",")
df2 = pd.DataFrame(f2)

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

logistic_regression(X_Train, Y_Train, X_Test, Y_Test)
random_forest(X_Train, Y_Train, X_Test, Y_Test)
cart(X_Train, Y_Train, X_Test, Y_Test)

dataset = []

for i in range(1,18):
    dataset.append(i)

print (precision_lr, precision_rf, precision_ct)

plt.plot(dataset, precision_lr, 'r', label='Untuned Logistic Regression')
plt.plot(dataset, precision_rf, 'g', label='Untuned Random Forest')
plt.plot(dataset, precision_ct, 'y', label='Untuned CART')
plt.show()