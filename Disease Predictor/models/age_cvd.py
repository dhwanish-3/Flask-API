import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

df=pd.read_csv('https://raw.githubusercontent.com/shellyannissa/Compute-Disease-Probability/main/cardio_train.csv',sep=';')
out=np.array(df)
x=out[:,12]==1
r=out[x]
x=r[:,2:12]
y=r[:,1]
y=y//365

model=LogisticRegression(max_iter=10000)
x_tr,x_te,y_tr,y_te=train_test_split(x,y,test_size=0.3,random_state=42)
'''
model.fit(x_tr,y_tr)
y_pred=model.predict(x_te)
corr_coef = np.corrcoef(y_pred,y_te)[0, 1]
corr_coef =0.084223577
'''
model.fit(x,y)

pickle.dump(model,open('age_cvd.pkl','wb'))