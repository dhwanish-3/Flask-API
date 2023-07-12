import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
import pickle 

df=pd.read_csv('https://raw.githubusercontent.com/shellyannissa/Compute-Disease-Probability/main/chronic%20kidney%20disease.csv')

df_age=pd.read_csv('https://raw.githubusercontent.com/shellyannissa/Compute-Disease-Probability/main/kidney_disease.csv')
df['age']=df_age['age']
df=df.drop(df[df['Class']==0].index)
df.drop('Class',axis=1,inplace=True)
df['age'].fillna(df['age'].median(),inplace=True)
df['age'].fillna(40,inplace=True)


x=df.iloc[:,0:13].values
y=df.iloc[:,13]
y=y.astype(int)
x_tr,x_te,y_tr,y_te=train_test_split(x,y,test_size=0.2,random_state=42)

model=GradientBoostingRegressor(n_estimators=100,learning_rate=0.1,max_depth=3)
model.fit(x_tr,y_tr)
y_pr=model.predict(x_te)
y_pr=y_pr.astype(int)


'''
calculating accuracy ,commenting to prevent from being run everytime model is loaded

def accur(y_true, y_pred, margin=10):
    # Calculate the absolute difference between the true and predicted values
    diff = np.abs(y_true - y_pred)  
    # Count the number of values that differ by less than or equal to the margin
    num_correct = np.sum(diff <= margin)
    acc = num_correct / len(y_true)
    return acc


acc=accur(y_te,y_pr)
accuracy of 0.6 with a margin of error of 10

'''
model.fit(x,y)

pickle.dump(model,open('age_chr_kid_dis.pkl','wb'))