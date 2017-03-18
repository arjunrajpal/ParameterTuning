from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score,accuracy_score
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
f1 = pd.read_csv("../final_dataset/ant/1.csv", delimiter = ",")
df1 = pd.DataFrame(f1)

f2 = pd.read_csv("../final_dataset/ant/3.csv", delimiter = ",")
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

f1 = pd.read_csv("../final_dataset/ant/2.csv", delimiter = ",")
df1 = pd.DataFrame(f1)

f2 = pd.read_csv("../final_dataset/ant/4.csv", delimiter = ",")
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

f1 = pd.read_csv("../final_dataset/ant/3.csv", delimiter = ",")
df1 = pd.DataFrame(f1)

f2 = pd.read_csv("../final_dataset/ant/5.csv", delimiter = ",")
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

f1 = pd.read_csv("../final_dataset/camel/1.csv", delimiter = ",")
df1 = pd.DataFrame(f1)

f2 = pd.read_csv("../final_dataset/camel/3.csv", delimiter = ",")
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

f1 = pd.read_csv("../final_dataset/camel/2.csv", delimiter = ",")
df1 = pd.DataFrame(f1)

f2 = pd.read_csv("../final_dataset/camel/4.csv", delimiter = ",")
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

f1 = pd.read_csv("../final_dataset/ivy/1.csv", delimiter = ",")
df1 = pd.DataFrame(f1)

f2 = pd.read_csv("../final_dataset/ivy/3.csv", delimiter = ",")
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

f1 = pd.read_csv("../final_dataset/jedit/1.csv", delimiter = ",")
df1 = pd.DataFrame(f1)

f2 = pd.read_csv("../final_dataset/jedit/3.csv", delimiter = ",")
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

f1 = pd.read_csv("../final_dataset/jedit/2.csv", delimiter = ",")
df1 = pd.DataFrame(f1)

f2 = pd.read_csv("../final_dataset/jedit/4.csv", delimiter = ",")
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

f1 = pd.read_csv("../final_dataset/jedit/3.csv", delimiter = ",")
df1 = pd.DataFrame(f1)

f2 = pd.read_csv("../final_dataset/jedit/5.csv", delimiter = ",")
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

f1 = pd.read_csv("../final_dataset/log4j/1.csv", delimiter = ",")
df1 = pd.DataFrame(f1)

f2 = pd.read_csv("../final_dataset/log4j/3.csv", delimiter = ",")
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

f1 = pd.read_csv("../final_dataset/lucene/1.csv", delimiter = ",")
df1 = pd.DataFrame(f1)

f2 = pd.read_csv("../final_dataset/lucene/3.csv", delimiter = ",")
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

f1 = pd.read_csv("../final_dataset/poi/1.csv", delimiter = ",")
df1 = pd.DataFrame(f1)

f2 = pd.read_csv("../final_dataset/poi/3.csv", delimiter = ",")
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

f1 = pd.read_csv("../final_dataset/poi/2.csv", delimiter = ",")
df1 = pd.DataFrame(f1)

f2 = pd.read_csv("../final_dataset/poi/4.csv", delimiter = ",")
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

f1 = pd.read_csv("../final_dataset/synapse/1.csv", delimiter = ",")
df1 = pd.DataFrame(f1)

f2 = pd.read_csv("../final_dataset/synapse/3.csv", delimiter = ",")
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

f1 = pd.read_csv("../final_dataset/velocity/1.csv", delimiter = ",")
df1 = pd.DataFrame(f1)

f2 = pd.read_csv("../final_dataset/velocity/3.csv", delimiter = ",")
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

f1 = pd.read_csv("../final_dataset/xerces/1.csv", delimiter = ",")
df1 = pd.DataFrame(f1)

f2 = pd.read_csv("../final_dataset/xerces/3.csv", delimiter = ",")
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

f1 = pd.read_csv("../final_dataset/xerces/2.csv", delimiter = ",")
df1 = pd.DataFrame(f1)

f2 = pd.read_csv("../final_dataset/xerces/4.csv", delimiter = ",")
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

print precision_lr, precision_rf, precision_ct

plt.plot(dataset, precision_lr, 'r', label='Untuned Logistic Regression')
plt.plot(dataset, precision_rf, 'g', label='Untuned Random Forest')
plt.plot(dataset, precision_ct, 'y', label='Untuned CART')
plt.show()