import csv
from sklearn import linear_model
import numpy as np

with open('../1.csv','r') as f:
    reader = csv.reader(f)
    data = list(reader)

data=np.asarray(data)
data=np.delete(data,[0,1],1)

data_withoutname=np.asarray(data[1:,1:]).astype(float)

X_train=data_withoutname[0:15,0:-1]
Y_train=data_withoutname[0:15,-1]


with open('../3.csv','r') as f:
    reader = csv.reader(f)
    data = list(reader)

data=np.asarray(data)
data=np.delete(data,[0,1],1)

data_withoutname=np.asarray(data[1:,1:]).astype(float)

X_test=data_withoutname[0:31,0:-1]
Y_test=data_withoutname[0:31,-1]

logistics=linear_model.LogisticRegression(multi_class='ovr')
logistics.fit(X_train,Y_train)

print "Predicted Values : ",logistics.predict(X_test)

print "Precision : ",logistics.score(X_test,Y_test)