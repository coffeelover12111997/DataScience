# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 14:34:42 2017

@author: preetish
"""

#quantify

import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.ensemble import  GradientBoostingRegressor,RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.linear_model import ElasticNet


train=pd.read_csv('C:/Users/preetish/Downloads/gcTrianingSet.csv')
test=pd.read_csv('C:/Users/preetish/Downloads/gcPredictionFile.csv')

train['cpufreq']=np.sqrt(np.log(1/train['cpuTimeTaken']))
train['subr']=train['initialUsedMemory']-train['initialFreeMemory']
train['tot']=train['initialUsedMemory']+train['initialFreeMemory']
y1=train['gcRun'].astype(int)
y2=train['finalFreeMemory']
y3=train['finalUsedMemory']
X=train.drop(['gcRun','finalFreeMemory','finalUsedMemory','cpuTimeTaken'],axis=1)



temp=pd.get_dummies(X['query token'])#for token vectors(train)

temp2=pd.get_dummies(test['query token'])

from sklearn.decomposition import PCA

pca=PCA(n_components=22)

temp=pca.fit_transform(temp)
temp2=pca.transform(temp2)

X=X.drop(['query token','userTime','sysTime','realTime','gcInitialMemory','gcFinalMemory','gcTotalMemory'],axis=1)


X=np.array(X)
X=np.concatenate((X,temp),axis=1)

test=test.drop('query token',axis=1)

test=np.array(test)
test=np.concatenate((test,temp2),axis=1)


trainx,cvx,trainy1,cvy1=train_test_split(X,y2,test_size=0.33,random_state=42)
trainx2,cvx2,trainy2,cvy2=train_test_split(X,y3,test_size=0.33,random_state=42)



model1=SVR(C=0.35,gamma=0.45)

model1.fit(trainx,trainy1)
print(model1.score(cvx,cvy1))
print(model1.score(trainx,trainy1))

m1=model1.predict(cvx)
t1=model1.predict(trainx)

model2=GradientBoostingRegressor(n_estimators=100,learning_rate=0.035,max_depth=3,min_samples_split=20)
model2.fit(trainx,trainy1)
print(model2.score(trainx,trainy1))
print(model2.score(cvx,cvy1))
t2=model2.predict(trainx)
m2=model2.predict(cvx)

m=np.sqrt(m1*m2)



model3=RandomForestRegressor(n_estimators=150,min_samples_split=3,max_leaf_nodes=300)
model3.fit(trainx2,trainy2)
print(model3.score(cvx2,cvy2))



























