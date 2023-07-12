import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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

out=np.array(df)
x=out[:,10]==1
r=out[x]
x=r[:,1:10]
y=r[:,0]
arr=df.iloc[1,1:10]

# model=LogisticRegression(max_iter=10000)
model=RandomForestClassifier(n_estimators=500,random_state=0)
# model.fit(x,y)
'''
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
corr_coef = np.corrcoef(y_pred,y_test)[0, 1]
corr_coef of 0.43267654799548627
'''
model.fit(x,y)

pickle.dump(model,open('age_stroke.pkl','wb'))