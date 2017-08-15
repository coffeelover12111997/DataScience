'''loan predictions'''

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder,scale
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from plot_learning_curve import plot_learning_curve
from time import time





if __name__=='__main__':

    '''loading data'''
    data=pd.read_csv('loantrain.csv')
    test=pd.read_csv('loantest.csv')

    '''handling null values'''
    for i in data.columns:
        if data[i].isnull().sum()>(data.shape[0])/2:
            data=data.drop(i,axis=1)

    for j in data.columns:
        if data[j].dtype=='int64' or data[j].dtype=='float64':
            data[j]=data[j].fillna(data[j].median())
        else:
            data[j]=data[j].fillna(np.array(data[j].mode())[0])

    for i in test.columns:
        if test[i].isnull().sum()>(test.shape[0])/2:
            test=test.drop(i,axis=1)

    for j in test.columns:
        if test[j].dtype=='int64' or test[j].dtype=='float64':
            test[j]=test[j].fillna(data[j].median())
        else:
            test[j]=test[j].fillna(np.array(data[j].mode())[0])

    '''after univariate analysis'''

    '''removing skewness'''
    data['Totinc']=data['ApplicantIncome']+data['CoapplicantIncome']
    data['ApplicantIncome']=np.log(data['ApplicantIncome']+10)
    data['CoapplicantIncome']=np.log(data['CoapplicantIncome']+10)
    data['LoanAmount']=np.log(data['LoanAmount']+10)
    data['Totinc']=np.log(data['Totinc']+10)

    test['Totinc']=test['ApplicantIncome']+test['CoapplicantIncome']
    test['ApplicantIncome']=np.log(test['ApplicantIncome']+10)
    test['CoapplicantIncome']=np.log(test['CoapplicantIncome']+10)
    test['LoanAmount']=np.log(test['LoanAmount']+10)
    test['Totinc']=np.log(test['Totinc']+10)
    
    data[['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term']]=scale( data[['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term']])
    test[['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term']]=scale( test[['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term']])
    #use scaler
    '''converting all kinds of data into numeric form'''
    
    colname=list(data.columns)
    colname.remove('Loan_Status')
    colname.remove('Loan_ID')
    X=pd.get_dummies(data[colname])
    test1=pd.get_dummies(test[colname])
    X=X.drop(['Dependents_1','Dependents_2','Dependents_3+','Married_Yes','Gender_Female','Married_No','Education_Not Graduate','Self_Employed_No'],axis=1)#these are not required
    test1=test1.drop(['Dependents_1','Dependents_2','Dependents_3+','Married_Yes','Gender_Female','Married_No','Education_Not Graduate','Self_Employed_No'],axis=1)
    encode1=LabelEncoder()
    y=encode1.fit_transform(data['Loan_Status'])

    trainx,testx,trainy,testy=train_test_split(X,y,random_state=42,test_size=0.2)#splitting my data into trainand cross validation set

    '''model selection'''
    t1=time()
    #model1=RandomForestClassifier(n_estimators=100,min_samples_split=20,max_depth=10,min_samples_leaf=10)
    #model1=SVC(C=1,gamma=0.05)
    #model1=GradientBoostingClassifier(n_estimators=100,learning_rate=0.01)
    
    model1.fit(trainx,trainy)
    print(time()-t1)
    print(model1.score(testx,testy))

    fig=plot_learning_curve(model1,'gbm',trainx,trainy.astype(int))
    fig.show()

    predictions=model1.predict(test1)
    predictions=predictions.astype(str)
    predictions[(predictions=='1')]='Y'
    predictions[predictions=='0']='N'
    sub=pd.DataFrame({
        'Loan_ID':test['Loan_ID'],
        'Loan_Status':predictions
                       })

    sub.to_csv('av.csv',index=False)
    

    

    
    

    
    
    
