import pandas as pd
import pickle

df = pd.read_csv("https://raw.githubusercontent.com/shellyannissa/Compute-Disease-Probability/main/survey%20lung%20cancer.csv")
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df['LUNG_CANCER']=encoder.fit_transform(df['LUNG_CANCER'])
df['GENDER']=encoder.fit_transform(df['GENDER'])
x=df.iloc[:,0:15]
y=df.iloc[:,15]
from sklearn.ensemble import RandomForestClassifier
'''
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
x_tr,x_te,y_tr,y_te=train_test_split(x,y,test_size=0.1,random_state=0)
model=RandomForestClassifier(n_estimators=200,random_state=0)
model.fit(x_tr,y_tr)
y_pred=model.predict(x_te)
acc=accuracy_score(y_pred,y_te)
accuracy of 0.967741935483871
'''
model=RandomForestClassifier(n_estimators=200,random_state=0)
model.fit(x,y)

pickle.dump(model,open('lung_cancer.pkl','wb'))