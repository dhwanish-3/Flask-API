from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle 

df=pd.read_csv('https://raw.githubusercontent.com/shellyannissa/Compute-Disease-Probability/main/Parkinsson%20disease.csv')
df.drop('name',axis=1,inplace=True)
col=list(df.columns)
col.remove('status')

'''
#Logistic Regression accuracy=89.74358974358975,corr=61.72133998483676
x=df[col]

y=df['status']
lg=LogisticRegression(max_iter=1000)
x_tr,x_te,y_tr,y_te=train_test_split(x,y,test_size=0.2,random_state=42)
lg.fit(x_tr,y_tr)
y_pred=lg.predict(x_te)
acc=accuracy_score(y_pred,y_te)
'''


'''
#GradienBoostingRegressor corr=79.16087078614259

from sklearn.ensemble import GradientBoostingRegressor
x=df[col]

y=df['status']
x_tr,x_te,y_tr,y_te=train_test_split(x,y,test_size=0.2,random_state=42)
gbr=GradientBoostingRegressor(n_estimators=100,learning_rate=0.1,max_depth=3)
gbr.fit(x_tr,y_tr)
y_pred=gbr.predict(x_te)
corr_coef = np.corrcoef(y_pred,y_te)[0, 1]

'''



x=df[col]


'''

#RandomForestClassifier with accuracy =94.87179487179487,corr=81.9920061690788,Best 
x_tr,x_te,y_tr,y_te=train_test_split(x,y,test_size=0.2,random_state=42)
rf=RandomForestClassifier(n_estimators=100,random_state=42)
rf.fit(x_tr,y_tr)
y_pred=rf.predict(x_te)
acc=accuracy_score(y_pred,y_te)
corr_coef = np.corrcoef(y_pred,y_te)[0, 1]
ft=rf.feature_importances_

'''
y=df['status']
req_col=['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)','MDVP:Flo(Hz)', 'MDVP:Jitter(Abs)','MDVP:RAP','Jitter:DDP','PPE','NHR', 'spread1','spread2']
df=df[req_col]

x=df.values


model=LogisticRegression(max_iter=1000)
'''
x_tr,x_te,y_tr,y_te=train_test_split(x,y,test_size=0.2,random_state=42)
lg.fit(x_tr,y_tr)
y_pred=lg.predict(x_te)
acc=accuracy_score(y_pred,y_te)
 accuracy of 0.9230769230769231
'''

model.fit(x,y)

pickle.dump(model,open('parkinsons.pkl','wb'))