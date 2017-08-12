'''Kaggle Housing challenge'''



'''libraries and modules imported for this problem'''
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn import metrics,linear_model,ensemble,svm
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import f_regression,SelectKBest

if __name__=='__main__':

    #loading data
    train=pd.read_csv('train.csv')
    test=pd.read_csv('test.csv')

    '''removing the features with excess null values''' 
    j=[]
    for i in train.columns:
        if train[i].isnull().sum()>700:
            j.append(i)
    train=train.drop(j,axis=1)
    test=test.drop(j,axis=1)


    '''filling remaining null values either with median or mode'''
    for i in train.columns:
        if train[i].dtype=='int64' or train[i].dtype=='float64':
            train[i]=train[i].fillna(train[i].median())
        else:
            train[i]=train[i].fillna(str(train[i].mode()))
    for i in test.columns:
        if test[i].dtype=='int64' or test[i].dtype=='float64':
            test[i]=test[i].fillna(train[i].median())
        else:
            test[i]=test[i].fillna(str(train[i].mode()))

    '''the below features were selected after checking correlation values and scatter plots'''
    l=['OverallQual', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'TotalBsmtSF', 'GrLivArea', 'FullBath', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars']
    
    '''transforming variables to get a normal distribution'''
    train['SalePrice']=np.log(train['SalePrice'])
    train['MasVnrArea']=np.log(train['MasVnrArea']+10)
    train['YearBuilt']=np.log(train['YearBuilt'])
    test['MasVnrArea']=np.log(test['MasVnrArea']+10)
    test['YearBuilt']=np.log(test['YearBuilt'])
    
   
    
    #pars={'C':np.arange(1,10000,1000)}
    #grid=GridSearchCV(model1,pars)
    
    '''trying out categorical features'''
    '''(finding categorical features and also removing those which are highly skewed)'''
    h=[]
    for i in train.columns:
        if train[i].dtype=='object':
            h.append(i)
    for j in h:
        if train[j].value_counts().max()>1000:
            h.remove(j)

    
    sel=SelectKBest(f_regression,5)
    data=train[h].copy()
    data=pd.get_dummies(data)
    data=data.drop(['Utilities_NoSeWa', 'Condition2_RRAe', 'Condition2_RRAn', 'Condition2_RRNn', 'HouseStyle_2.5Fin', 'RoofMatl_ClyTile', 'RoofMatl_Membran', 'RoofMatl_Metal', 'RoofMatl_Roll', 'Exterior1st_ImStucc', 'Exterior1st_Stone', 'Exterior2nd_Other', 'Heating_Floor', 'Heating_OthW', 'Electrical_0    SBrkr\ndtype: object', 'Electrical_Mix', 'FireplaceQu_0    Gd\ndtype: object'],axis=1)

    testfeatures=[]
    sel.fit(data,train['SalePrice'])
    fr=sel.transform(data)
    fr=pd.DataFrame(fr)
    trainX=pd.concat([pd.DataFrame(scale(train[l])),fr],axis=1)
    
    X=pd.get_dummies(test[h]) 
    for x in data.columns:
        if x in X.columns:
            testfeatures.append(x)

    
    fr1=sel.transform(X[testfeatures])
    fr1=pd.DataFrame(fr1)
    testX=pd.concat([pd.DataFrame(scale(test[l])),fr1],axis=1)


    '''splitting into train and test'''
    train_X,test_X,train_y,test_y=train_test_split(trainX,train['SalePrice'],test_size=0.33,random_state=42)

    #hyperparameter tuning not done yet
    
    '''RandomForest'''
    model1=ensemble.RandomForestRegressor()
    model1.fit(train_X,train_y)
    print(model1.score(test_X,test_y))

   '''Svm'''
    model2=svm.SVR(C=2.5)
    model2.fit(train_X,train_y)
    print(model2.score(test_X,test_y))
    
    





    '''#train=train.dropna(axis=1,thresh=700)
    trainX=train[['OverallQual','GrLivArea','GarageArea','1stFlrSF','FullBath','YearBuilt','TotRmsAbvGrd']]
    temp=train[['MSZoning','LotShape','LotConfig']]
    trainX['GrLivArea']=np.log(trainX['GrLivArea'])
    trainX['1stFlrSF']=np.log(trainX['1stFlrSF'])
    temp=pd.get_dummies(temp)
    temp=np.array(temp)
    trainX=scale(trainX)
    trainX=np.concatenate((trainX,temp),axis=1)
    trainy=np.log(train['SalePrice'])
    testX=test[['OverallQual','GrLivArea','GarageArea','1stFlrSF','FullBath','YearBuilt','TotRmsAbvGrd']]
    temp2=test[['MSZoning','LotShape','LotConfig']]
    testX['GarageArea']=testX['GarageArea'].fillna(testX['GarageArea'].mean())
    testX['GrLivArea']=np.log(testX['GrLivArea'])
    testX['1stFlrSF']=np.log(testX['1stFlrSF'])
    temp2=pd.get_dummies(temp2)
    testX=scale(testX)
    testX=np.concatenate((testX,temp2),axis=1)
    train_X,test_X,train_y,test_y=train_test_split(trainX,trainy,test_size=0.33,random_state=42)
    model1=linear_model.LinearRegression()
    model1.fit(train_X,train_y)
    print(metrics.r2_score(test_y,model1.predict(test_X)))
    '''

    prediction=model1.predict(testX)
    '''submission file'''
    sub=pd.DataFrame({
                   'Id':test['Id'],
                   'SalePrice':np.exp(prediction)})
    sub.to_csv('kaggle1.csv',index=False)
    
    
