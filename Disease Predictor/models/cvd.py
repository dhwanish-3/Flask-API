import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

df=pd.read_csv('https://raw.githubusercontent.com/shellyannissa/Compute-Disease-Probability/main/cardio_train.csv',sep=';')
x=df.iloc[:,1:12]
y=df.iloc[:,12]
model=RandomForestClassifier(n_estimators=300,random_state=42)

'''
x_tr,x_te,y_tr,y_te=train_test_split(x,y,test_size=0.3,random_state=42)
model.fit(x_tr,y_tr)
y_pred=model.predict(x_te)
acc=accuracy_score(y_pred,y_te)
acc=0.7470952380952381
'''
model.fit(x,y)

pickle.dump(model,open('cvd.pkl','wb'))