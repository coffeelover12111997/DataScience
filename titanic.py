#this file contains cleaning and model selection and parameter tuning feature selection is in another file

import numpy as np              #for numerical computations and vector usage
import pandas as pd             #for loading the data and using dataframes
import matplotlib.pyplot as plt #for plotting 
from sklearn import linear_model #for logisticregression
from sklearn import ensemble    #for randomforests,baggingtrees,gbm
from sklearn import model_selection
from sklearn import svm,tree    #for SVC and decision trees
from plot_learning_curve import plot_learning_curve #this has the code to plot learning curves
from sklearn import metrics     #for crossvalidation
from sklearn.preprocessing import Imputer,scale #for datacleaning and preprocessing
from sklearn.model_selection import train_test_split #for splitting data(stratified K fold with k=10 is also fine)
import warnings                 #to prevent warnings
from time import time           #to check training time and testing time

if __name__=='__main__':
    warnings.filterwarnings("ignore")
    
    '''Loading The Training And Test data'''
    data=pd.read_csv('train.csv')
    test=pd.read_csv('test.csv')
    
    '''data cleaning'''
    data['Embarked']=data['Embarked'].fillna('S')
    trainX=data[['Pclass','Age','SibSp','Parch','Fare']] #trainX is the dataframe using for training the model
    testX=test[['Pclass','Age','SibSp','Parch','Fare']]  #testX is used for prediction 
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
    temp2=test[['Sex','Embarked']]
    temp2=pd.get_dummies(temp2)
    testX=scale(testX)
    testX=np.concatenate((testX,temp2),axis=1)
    
    '''splitting into training and validation sets'''
    #train_X,cv_X,train_y,cv_y=train_test_split(trainX,trainy,random_state=42)
    skf=model_selection.StratifiedKFold(n_splits=10) #using startifiedKfold to use the above just comment it out and comment this section
    for tr_index,te_index in skf.split(trainX,trainy):
        train_X,cv_X=trainX[tr_index],trainX[te_index]
        train_y,cv_y=trainy[tr_index],trainy[te_index]
    
    
    
    
    '''using gbm and parameter tuning using gridsearch'''
    pars={'learning_rate':[.001],'n_estimators':list(range(1000,10000,100)),'max_depth':list(range(2,20)),'min_samples_split':list(range(3,20))}
    model1=ensemble.GradientBoostingClassifier()
    grid=model_selection.GridSearchCV(model1,pars)
    t1=time()
    grid.fit(train_X,train_y.astype(int))
    print(metrics.accuracy_score(cv_y.astype(int),grid.predict(cv_X)))
   
    
    '''using svm'''
    model3=svm.SVC(C=1.25)
    #t0=time()
    model3.fit(train_X,train_y.astype(int))
    #fig=plot_learning_curve(svm.SVC(),'svm',trainX,trainy.astype(int)) #(to plot learning curve comment this line out) 
    #fig.show()
    print("training time0:", round(time()-t0, 3), "s")
    print(metrics.accuracy_score(cv_y.astype(int),model3.predict(cv_X))) #to check accuracy of our model
    
    
    

    
    '''submitting test predictions using the best model'''
    predictions=model3.predict(testX)
    sub=pd.DataFrame({
        'PassengerId':test['PassengerId'],
        'Survived':predictions
                       })
    sub.to_csv("kaggle.csv", index=False)
    
