import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
import pickle



if __name__=="__main__":

    df=pd.read_csv("Covid-19(clss).csv")
    df['Gender']=pd.get_dummies(df['Gender'])
    
    y=df[['Victims']]
    X=df.drop('Victims',axis=1)


    x_train,x_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

    x_train=x_train.to_numpy()
    x_test=x_test.to_numpy()

    y_train=y_train.to_numpy().reshape(2260,)
    y_test=y_test.to_numpy().reshape(566,)

    clf=LogisticRegression()
    clf.fit(x_train,y_train)

    file=open('model.pkl','wb')
    pickle.dump(clf, file)
    file.close()

    #code for predic on given input 
    # var=[[1,1,101,0,1,1,0,22,0,0]]
    # ans=clf.predict(var)
    # print(ans)
    # print(clf.predict_proba(var)[0][1])
    