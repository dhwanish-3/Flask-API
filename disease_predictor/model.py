import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pickle

symp=pd.read_csv('https://raw.githubusercontent.com/shellyannissa/Compute-Disease-Probability/main/Symptom-severity.csv')
df=pd.read_csv('https://raw.githubusercontent.com/shellyannissa/Compute-Disease-Probability/main/dataset.csv')
symptoms=symp['Symptom']
symptoms=sorted(symptoms)
train=pd.DataFrame()
t=df.drop(columns='Disease')
#creating a hot set code dataframe
train=pd.get_dummies(t.stack()).sum(level=0)
# train=train.sort_index(axis=1)
train=train.reindex(sorted(train.columns),axis=1)

#removing leading spaces
new_col={col:col.strip()for col in train.columns}
train=train.rename(columns=new_col)

train['Prognosis']=df['Disease']
train=train.sort_index(axis=1)
train=train.reset_index(drop=True)
x=train.iloc[:,1:132].values
y=train.iloc[:,0]

model=LogisticRegression()
model.fit(x,y)
'''
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
accuracy of 100 percent
'''

pickle.dump(model,open('symptoms.pkl','wb'))