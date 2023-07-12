import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pickle

df=pd.read_csv('https://raw.githubusercontent.com/shellyannissa/Compute-Disease-Probability/main/healthcare-dataset-stroke-data.csv')
Male=pd.get_dummies(df['gender'])
Male.drop('Female',axis=1,inplace=True)
Male.drop('Other',axis=1,inplace=True)
df['Male']=Male
df.drop('gender',axis=1,inplace=True)
Mar=pd.get_dummies(df['ever_married'])
Mar.drop('No',axis=1,inplace=True)
df['Married']=Mar
df.drop('ever_married',axis=1,inplace=True)
Urb=pd.get_dummies(df['Residence_type'])
Urb.drop('Rural',axis=1,inplace=True)
df['Urb']=Urb
df.drop('Residence_type',axis=1,inplace=True)
smoker_type=pd.get_dummies(df['smoking_status'])
smoker_type.rename(columns={'Unknown':'unkown_smok_st'},inplace=True)
df=pd.concat((df,smoker_type),axis=1)
df.drop('smoking_status',axis=1,inplace=True)
work_type=pd.get_dummies(df['work_type'])
work_type.drop('children',axis=1,inplace=True)
df=pd.concat((df,work_type),axis=1)
df.drop('work_type',axis=1,inplace=True)
df.dropna(inplace=True)
df.drop(['unkown_smok_st','formerly smoked','never smoked','Private'],axis=1,inplace=True)
df['working_st']=df['Govt_job']+df['Self-employed']
df.drop(['Govt_job','Self-employed'],axis=1,inplace=True)
new_order=[col for col in df.columns if col!='stroke']+['stroke']
df=df.reindex(columns=new_order)
df.drop('id',axis=1,inplace=True)
df.drop('Never_worked',axis=1,inplace=True)
model=LogisticRegression(max_iter=10000)
x=df.iloc[:,0:10].values
y=df.iloc[:,10]
'''
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
accuracy of 0.9562118126272913
'''
model.fit(x,y)

pickle.dump(model,open('stroke.pkl','wb'))