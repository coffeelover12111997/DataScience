import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale,LabelEncoder
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_selection import SelectKBest
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.optimizers import Adagrad


def dataclean(traindata):
     #traindata['mvar3'].replace(to_replace=0,value=np.nan,inplace=True)
    traindata['mvar9'].replace(to_replace=0,value=traindata['mvar9'].median(),inplace=True)
    
        

    traindata=traindata.drop(['mvar3','mvar1'],axis=1)#3,1,14,46,47,48(2,4)(var in brackets have no changes)
    traindata['mvar5']=np.log(traindata['mvar5']+10)#5
    traindata['mvar6']=np.sqrt(traindata['mvar6']+1)#6
    traindata['mvar7']=np.sqrt(traindata['mvar7']+1)#7
    traindata['mvar8']=np.log(traindata['mvar8']+0.1)#8(10,11,12,13)
    traindata['mvar9']=np.log(traindata['mvar9'])
    traindata['mvar16']=np.log(traindata['mvar16']+0.1)#16
    traindata['mvar17']=np.log(traindata['mvar17']+0.1)#17
    traindata['mvar18']=np.log(traindata['mvar18']+0.1)#18
    traindata['mvar19']=np.log(traindata['mvar19']+0.1)#19  
    traindata['mvar20']=np.log(traindata['mvar20']+1)#20
    traindata['mvar21']=np.log(traindata['mvar21']+1)#21
    traindata['mvar22']=np.log(traindata['mvar22']+1)#22
    traindata['mvar23']=np.log(traindata['mvar23']+1)#23
    traindata['mvar24']=np.log(traindata['mvar24']+1)#24
    traindata['mvar25']=np.log(traindata['mvar25']+1)#25
    traindata['mvar26']=np.log(traindata['mvar26']+1)#26
    traindata['mvar27']=np.log(traindata['mvar27']+1)#27
    traindata['mvar28']=np.log(traindata['mvar28']+0.1)#28
    traindata['mvar29']=np.log(traindata['mvar29']+0.1)#29
    traindata['mvar30']=np.log(traindata['mvar30']+0.1)#30
    traindata['mvar31']=np.log(traindata['mvar31']+0.1)#31
    #traindata['mvar32']=np.log(traindata['mvar32'].astype(float)+1)
    #traindata['mvar33']=np.sqrt(traindata['mvar33'].astype(float)+1)
    #traindata['mvar34']=np.sqrt(traindata['mvar34'].astype(float)+1)    
    #traindata['mvar35']=np.sqrt(traindata['mvar35'].astype(float)+1)
    traindata['mvar36']=np.log(traindata['mvar36']+0.1)#36    
    traindata['mvar37']=np.log(traindata['mvar37']+0.1)#37
    traindata['mvar38']=np.log(traindata['mvar38']+0.1)#38
    traindata['mvar39']=np.log(traindata['mvar39']+0.1)#39(40,41,42,43,44,45)
    

    
    #Scale_values
    '''
    #main.append('mvar2')
    #main.append('mvar4')'''
    
    
    '''total house hold spend'''
    traindata['24_corr']=(traindata['mvar24']+traindata['mvar25']+traindata['mvar26']+traindata['mvar27'])

    '''expenditure per person'''
    temp=((traindata['mvar36']+traindata['mvar37']+traindata['mvar38']+traindata['mvar39'])/4)
    traindata['perperson']=temp/(traindata['mvar2']+1)

    '''total card pay percent
    traindata['cpp']=traindata['mvar13']/temp'''

    '''club to total exp'''
    traindata['ctt']=traindata['mvar6']/temp

    '''inc to expend'''
    traindata['eti']=temp/traindata['mvar9']

    '''buss exp'''
    traindata['be']=traindata['mvar11']*temp*traindata['mvar7']

    '''fee per club'''
    traindata['fpc']=traindata['mvar6']/(traindata['mvar14']+1)

    
    

    

    
    

    

   

    
    
    #traindata['ec _corr']=(traindata['mvar16']+traindata['mvar17']+traindata['mvar18']+traindata['mvar19'])/4
    
    
    
    #traindata=traindata.drop(['mvar31','mvar30','mvar15','mvar14','mvar41','mvar42','mvar45','mvar44','mvar24','mvar25','mvar26','mvar27','mvar38'],axis=1)#(24,25,26,27),(28,29,30,31),(38,39) highly correlated

    return traindata













if __name__=='__main__':
    traindata=pd.read_csv('Training_Dataset_1.csv')
    LeaderBoard_Data=pd.read_csv('Final_Dataset.csv')
    '''data cleaning'''
    
    
    

    main=[]
    for i in traindata.columns:
        if traindata[i].dtype=='float64':
            main.append(i)
    

    remove=[]
    '''for j in main:
        traindata[j].replace(to_replace=0,value=np.nan,inplace=True)
        if traindata[j].isnull().sum()>(traindata.shape[0]/2):
            remove.append(j)
            traindata=traindata.drop(j,axis=1)
        else:
            traindata[j]=traindata[j].fillna(traindata[j].median())'''
    

    for k in remove:
        LeaderBoard_Data=LeaderBoard_Data.drop(k,axis=1)

    traindata=dataclean(traindata)
    LeaderBoard_Data=dataclean(LeaderBoard_Data)

    main=[]
    for i in traindata.columns:
        if traindata[i].dtype=='float64':
            main.append(i)

    traindata[main]=scale(traindata[main])
    LeaderBoard_Data[main]=scale(LeaderBoard_Data[main])


    

    tri=pd.get_dummies(traindata['mvar12'])
    dimr=PCA(n_components=2)
    tri=dimr.fit_transform(tri)
    tri=pd.DataFrame(tri)
    tri['cm_key']=traindata['cm_key']
    traindata=traindata.merge(tri)

    traindata=traindata.drop(['cm_key','mvar12'],axis=1)

    tri2=pd.get_dummies(LeaderBoard_Data['mvar12'])
    tri2=dimr.transform(tri2)
    tri2=pd.DataFrame(tri2)
    tri2['cm_key']=LeaderBoard_Data['cm_key']
    LeaderBoard_Data=LeaderBoard_Data.merge(tri2)
    
    
    lead_key=LeaderBoard_Data['cm_key'].copy()
    LeaderBoard_Data=LeaderBoard_Data.drop(['cm_key','mvar12'],axis=1)

    
    
    

    
    


    y=traindata[['mvar49','mvar50','mvar51']].copy()
    y1=traindata[['mvar46','mvar47','mvar48']].copy()
    X=traindata.drop(['mvar49','mvar50','mvar51','mvar46','mvar47','mvar48'],axis=1)

    oneX=[]
    oney=[]
    preX=[]
    prey=[]
    t1=np.array(X)
    tr=np.array(y)
    tt=np.array(y1)
    '''for i in range(y.shape[0]):
        if tr[i][0]==1 or tr[i][1]==1 or tr[i][2]==1:
            oneX.append(t1[i])
            oney.append(tr[i])


    for i in range(15000):
        if tr[i][0]==0 and tr[i][1]==0 and tr[i][2]==0:
            zeroX.append(t1[i])
            zeroy.append(tt[i])'''

    for i in range(y.shape[0]):
        if tr[i][0]==1 :
            oneX.append(t1[i])
            oney.append(0)
        elif tr[i][1]==1 :
            oneX.append(t1[i])
            oney.append(1)
        elif tr[i][2]==1 :
            oneX.append(t1[i])
            oney.append(2)
        

    for j in range(20000):
        if tr[j][0]==0 and tr[j][1]==0 and tr[j][2]==0:
            if tt[j][0]==1:
                oneX.append(t1[j])
                oney.append(3)
            elif tt[j][1]==1:
                oneX.append(t1[j])
                oney.append(4)
            elif tt[j][2]==1:
                oneX.append(t1[j])
                oney.append(5)
                


   
        


    
    
    
    '''part2'''    
           

    #finalX=np.concatenate((oneX,zeroX),axis=0)
    #finaly=np.concatenate((oney,zeroy),axis=0)

    





    oneX=np.array(oneX)
    oney=np.array(oney)

    oney = np.array(pd.get_dummies(pd.Series(oney)))

    
    
    


    




    #data=np.array(LeaderBoard_Data)
    key=np.array(lead_key)

    data=np.array(LeaderBoard_Data)
    
    




    n=30
    ag=Adagrad(lr=0.01,epsilon=1e-08,decay=0.0)
    Classifier = Sequential()
    Classifier.add(Dense(output_dim= n , init = 'uniform',activation = 'relu',input_dim = oneX.shape[1]))
    Classifier.add(Dropout(0.1))
    Classifier.add(Dense(output_dim= n , init = 'uniform',activation = 'relu'))
    Classifier.add(Dropout(0.1))
    Classifier.add(Dense(output_dim= 6 , init = 'uniform',activation = 'sigmoid'))
    Classifier.compile(optimizer = ag ,loss = 'categorical_crossentropy',metrics=['accuracy'])
    Classifier.fit(oneX,oney, batch_size=10, nb_epoch = 100)

    
    
   
    submit=Classifier.predict(data)
    
    

    submit1=[]
    m=0.508715
    w=1.4
    for i in range(submit.shape[0]):
        
        q=np.argmax(submit[i])
        
        if q==3:
            submit1.append(q)
        elif q==4:
            submit1.append(q)
        elif q==5:
            submit1.append(q)
        if submit[i][3]+submit[i][4]+submit[i][5]<w:
            if q==0:
                if submit[i][q]>m:
                    submit1.append(q)
                else:
                    submit1.append(3)
            elif q==1:
                if submit[i][q]>m:
                    submit1.append(q)
                else:
                    submit1.append(3)
            elif q==2:
                if submit[i][q]>m:
                    submit1.append(q)
                else:
                    submit1.append(3)
        else:
            submit1.append(3)
    
    index=[]
    amex=[]

    k=0
    l=0
    while l<10000:
       if submit1[l]==0 or submit1[l]==1 or submit1[l]==2:
           index.append(l)
           k+=1
           l+=1
       else:
           l+=1
           continue

    
        
        



    print(k)
    for i in index:
        if submit1[i]==2:
            amex.append('Credit')
        elif submit1[i]==0:
            amex.append('Supp')
        else:
            amex.append('Elite')


    df=pd.DataFrame({
        'key':key[index],'type':np.array(amex)})
    df.to_csv('adversials_iitkgp_4.csv',index=False)
    
    
    
    
    
    
