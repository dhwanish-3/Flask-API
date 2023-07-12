from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import pickle

df=pd.read_csv('https://raw.githubusercontent.com/shellyannissa/Compute-Disease-Probability/main/diabetes.csv')
x=np.array(df)
out=x[:,8]==1
r=x[out]
a=r[:,:7]
b=r[:,7]
model=RandomForestClassifier(n_estimators=500,random_state=0)
'''
x_tr,x_te,y_tr,y_te=train_test_split(a,b,test_size=0.3,random_state=42)
model.fit(x_tr,y_tr)
y_pred=model.predict(x_te)
corr_coef = np.corrcoef(y_pred,y_te)[0, 1]
corr_coef =0.6048295124464195
'''
model.fit(a,b)

pickle.dump(model,open('age_diab.pkl','wb'))