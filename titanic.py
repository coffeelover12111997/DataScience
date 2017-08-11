import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import ensemble
from dataclean import dclean
from fscale import fscale
from sklearn import model_selection
from sklearn import svm,tree,ensemble
from plot_learning_curve import plot_learning_curve
from sklearn import metrics
from sklearn.preprocessing import Imputer,scale
from sklearn.model_selection import train_test_split
import warnings
from time import time

if __name__=='__main__':
    warnings.filterwarnings("ignore")
    data=pd.read_csv('train.csv')
    test=pd.read_csv('test.csv')
    data['Embarked']=data['Embarked'].fillna('S')
    trainX=data[['Pclass','Age','SibSp','Parch','Fare']]
    testX=test[['Pclass','Age','SibSp','Parch','Fare']]
    trainX['Age']=trainX['Age'].fillna(trainX['Age'].median())
    testX['Age']=testX['Age'].fillna(testX['Age'].median())
    testX['Fare']=testX['Fare'].fillna(testX['Fare'].median())
    trainX['family']=pd.Series(np.array(trainX['SibSp'])+np.array(trainX['Parch']))
    testX['family']=pd.Series(np.array(testX['SibSp'])+np.array(testX['Parch']))
    trainX['Adult']=pd.Series(trainX['Age']>=21,dtype=int)
    testX['Adult']=pd.Series(testX['Age']>=21,dtype=int)
    temp=data[['Sex','Embarked']]
    temp=pd.get_dummies(temp)
    temp=np.array(temp)
    trainy=np.array(data['Survived'])
    trainX=scale(trainX)
    trainX=np.concatenate((trainX,temp),axis=1)
    #train_X,cv_X,train_y,cv_y=train_test_split(trainX,trainy,random_state=42)
    temp2=test[['Sex','Embarked']]
    temp2=pd.get_dummies(temp2)
    testX=scale(testX)
    testX=np.concatenate((testX,temp2),axis=1)
    skf=model_selection.StratifiedKFold(n_splits=3)
    for tr_index,te_index in skf.split(trainX,trainy):
        train_X,cv_X=trainX[tr_index],trainX[te_index]
        train_y,cv_y=trainy[tr_index],trainy[te_index]
    #pars={'learning_rate':[.001],'n_estimators':list(range(1000,10000,100)),'max_depth':list(range(2,20)),'min_samples_split':list(range(3,20))}
    #model1=ensemble.GradientBoostingClassifier()
    #tri=model_selection.GridSearchCV(model1,pars)
    #t1=time()
    #model1.fit(train_X,train_y.astype(int))
    #model2=neural_network.MLPClassifier()
    #model2.fit(train_X,train_y.astype(int))
    #print("training time1:", round(time()-t1, 4), "s")
    model3=svm.SVC(C=1.25)
    #t0=time()
    model3.fit(train_X,train_y.astype(int))
    #fig=plot_learning_curve(svm.SVC(),'svm',trainX,trainy.astype(int))
    #fig.show()
    #print("training time0:", round(time()-t0, 3), "s")
    #fig=plot_learning_curve(svm.SVC(),'svm',trainX,trainy.astype(int))
    #fig.show()
    print(metrics.accuracy_score(cv_y.astype(int),model3.predict(cv_X)))
    #print(metrics.accuracy_score(cv_y.astype(int),model2.predict(cv_X)))
    #print(metrics.accuracy_score(cv_y.astype(int),model3.predict(cv_X)))

    

    predictions=model3.predict(testX)
    sub=pd.DataFrame({
        'PassengerId':test['PassengerId'],
        'Survived':predictions
                       })
    sub.to_csv("kaggle.csv", index=False)
    
