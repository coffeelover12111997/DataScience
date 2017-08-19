import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn import neighbors,decomposition,linear_model,multiclass,svm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

if __name__=='__main__':
    train=pd.read_csv('train.csv')
    test=pd.read_csv('test.csv')
    dimr=decomposition.PCA(n_components=25,svd_solver='randomized',whiten=True)
    cpy=train.copy()
    labels=train['label']
    features=cpy.drop('label',axis=1)
    dimr.fit(features)
    newfeature=dimr.transform(features)
    testfeat=dimr.transform(test)
    #nb={'n_neighbors':range(5,20,5)}
    #c={'C':range(1,1000,100)}
    #trainX,testX,trainy,testy=train_test_split(newfeature,labels,test_size=0.33,random_state=42)
    knn=neighbors.KNeighborsClassifier(n_neighbors=12)
    #model1=GridSearchCV(knn,nb)
    knn.fit(newfeature,labels)
    #print(knn.score(testX,testy))

    #Lr=linear_model.LogisticRegression(penalty='l2')
    #sv=svm.SVC()
    #model2=GridSearchCV(Lr,c)
    #mod2=multiclass.OneVsRestClassifier(sv)
    #l2=pd.get_dummies(labels)
    #mod2.fit(features,labels)
    
    
    answers=knn.predict(testfeat)
    df=pd.DataFrame({
        'ImageId':range(1,answers.shape[0]+1),'Label':answers})
    df.to_csv('kaggle.csv',index=False)
    
    
    
