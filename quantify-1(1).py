# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 14:34:42 2017

@author: preetish
"""

#Goldman Sachs quantify

import numpy as np
import pandas as pd
from sklearn.svm import SVR, SVC
from sklearn.model_selection import train_test_split
from sklearn.ensemble import  GradientBoostingRegressor, RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, recall_score
from sklearn.neighbors import KNeighborsClassifier

train=pd.read_csv('C:/Users/preetish/Downloads/gcTrianingSet.csv')
test=pd.read_csv('C:/Users/preetish/Downloads/gcPredictionFile.csv')

test['cpufreq']=np.sqrt(np.log(1/test['cpuTimeTaken']))
test['subr']=test['initialUsedMemory']-test['initialFreeMemory']
test['tot']=test['initialUsedMemory']+test['initialFreeMemory']

train['cpufreq']=np.sqrt(np.log(1/train['cpuTimeTaken']))
train['subr']=train['initialUsedMemory']-train['initialFreeMemory']
train['tot']=train['initialUsedMemory']+train['initialFreeMemory']

y1=train['gcRun'].astype(int)
y2=train['finalFreeMemory']
y3=train['finalUsedMemory']
X=train.drop(['gcRun','finalFreeMemory','finalUsedMemory','cpuTimeTaken'],axis=1)
ytest = test['gcRun']
xtest = test.drop(['gcRun','cpuTimeTaken', 'query token'],1)

temp=pd.get_dummies(X['query token'])#for token vectors(train)

temp2=pd.get_dummies(test['query token'])

from sklearn.decomposition import PCA

pca=PCA(n_components=22)

temp=pca.fit_transform(temp)
temp2=pca.transform(temp2)

X=X.drop(['query token','userTime','sysTime','realTime','gcInitialMemory','gcFinalMemory','gcTotalMemory'],axis=1)


X=np.array(X)
X=np.concatenate((X,temp),axis=1)

#test=test.drop('query token',axis=1)

xtest=np.array(xtest)
xtest=np.concatenate((xtest,temp2),axis=1)

trainx,cvx,trainy1,cvy1=train_test_split(X,y2,test_size=0.33,random_state=42)
trainx2,cvx2,trainy2,cvy2=train_test_split(X,y3,test_size=0.33,random_state=42)

model1=SVR(C=0.25,gamma=0.4)

model1.fit(trainx,trainy1)
print(model1.score(trainx,trainy1))
m1=model1.predict(cvx)
t1=model1.predict(trainx)

model2=GradientBoostingRegressor(n_estimators=2000,learning_rate=0.005,max_depth=2,min_samples_split=35)
model2.fit(trainx,trainy1)
print(model2.score(trainx,trainy1))
t2=model2.predict(trainx)
m2=model2.predict(cvx)

m=np.sqrt(m1*m2)

model3 = RandomForestRegressor(n_estimators = 150, max_leaf_nodes = 300, min_samples_split = 3)
model3.fit(trainx2, trainy2)

for i in range(1624):
    xtest[i+1,1] = np.sqrt(model1.predict(xtest[i,:].reshape(1,-1))*model2.predict(xtest[i,:]).reshape(1,-1))
    xtest[i+1,0] = model3.predict(xtest[i,:].reshape(1,-1))
    xtest[i+1,3]=xtest[i+1,0]-xtest[i+1,1]
    xtest[i+1,4]=xtest[i+1,0]+xtest[i+1,1]
    
trainccx, testcx, traincy, testcy = train_test_split(X,y1,test_size=0.75,random_state=42)

classmodel = RandomForestClassifier(n_estimators = 500, max_depth = 10, min_samples_split = 5)
classmodel.fit(trainccx,traincy)
print(classmodel.score(trainccx,traincy))
print(classmodel.score(testcx,testcy))
print(recall_score(traincy,classmodel.predict(trainccx)))
print(recall_score(testcy,classmodel.predict(testcx)))

cm1 = SVC(C = 500, gamma = 0.1)
cm1.fit(trainccx,traincy)
print(recall_score(traincy,cm1.predict(trainccx)))
print(recall_score(testcy,cm1.predict(testcx)))

cm2 = KNeighborsClassifier(n_neighbors = 3)
cm2.fit(trainccx,traincy)
print(recall_score(traincy,cm2.predict(trainccx)))
print(recall_score(testcy,cm2.predict(testcx)))


    

final = np.ones([1625,2])
final[:,0] = xtest[:,1]
#final[:,1] = cm2.predict(xtest)

for i in range(1624):
    if(cm2.predict(testcx)[i]+classmodel.predict(testcx)[i]+cm1.predict(testcx)[i]>1):
        final[i,1] =1
    else:
        final[i,1] = 0
        

sub = pd.DataFrame({
                    'serialNum': np.arange(1,1626),
                    'initialFreeMemory': final[:,0],
                    'gcRun': final[:,1].astype(bool)})
    
sub.to_csv("Pheonix_kgp2.csv", index = False)




















