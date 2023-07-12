import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pickle

df=pd.read_csv('https://raw.githubusercontent.com/shellyannissa/Compute-Disease-Probability/main/chronic%20kidney%20disease.csv')

x=df.iloc[:,0:13].values
y=df.iloc[:,13]
y=y.astype(int)
model=LogisticRegression(max_iter=1000)
'''
x_tr,x_te,y_tr,y_te=train_test_split(x,y,test_size=0.2,random_state=42)
model.fit(x_tr,y_tr)
y_pred=model.predict(x_te)
acc=accuracy_score(y_te,y_pred)
accuracy of 0.975
'''

model.fit(x,y)

pickle.dump(model,open('chr_kid_dis.pkl','wb'))