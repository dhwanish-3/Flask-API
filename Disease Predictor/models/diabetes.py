import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

df=pd.read_csv('https://raw.githubusercontent.com/shellyannissa/Compute-Disease-Probability/main/diabetes.csv')
x=df.iloc[:,0:8].values
y=df.iloc[:,8].values
model=RandomForestClassifier(n_estimators=200,random_state=0)
'''
x_tr,x_te,y_tr,y_te=train_test_split(x,y,test_size=0.1,random_state=0)
model.fit(x_tr,y_tr)
y_pred=model.predict(x_te)
acc=accuracy_score(y_pred,y_te)
accuracy of 0.8051948051948052
'''
model.fit(x,y)

pickle.dump(model,open('diab.pkl','wb'))