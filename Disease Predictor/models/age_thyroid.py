import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pickle

df=pd.read_csv('https://raw.githubusercontent.com/shellyannissa/Compute-Disease-Probability/main/hypothyroid.csv')

df=df.replace({'t':1,'f':0})
df=df.replace({'?':np.NAN})
df=df.replace({'F':1,'M':0})
df=df.replace({'P':1,'N':0})
df['T4U measured'].fillna(df['T4U measured'].mean(),inplace=True)
df['sex'].fillna(df['sex'].mean(),inplace=True)
df['age']=pd.to_numeric(df['age'],errors='coerce')
df['age'].fillna(df['age'].mean(),inplace=True)
df['TSH']=pd.to_numeric(df['TSH'],errors='coerce')
df['TSH'].fillna(df['TSH'].mean(),inplace=True)
df['T3']=pd.to_numeric(df['T3'],errors='coerce')
df['T3'].fillna(df['T3'].mean(),inplace=True)
df['TT4']=pd.to_numeric(df['TT4'],errors='coerce')
df['TT4'].fillna(df['TT4'].mean(),inplace=True)
df['T4U']=pd.to_numeric(df['T4U'],errors='coerce')
df['T4U'].fillna(df['T4U'].mean(),inplace=True)
df['FTI']=pd.to_numeric(df['FTI'],errors='coerce')
df['FTI'].fillna(df['FTI'].mean(),inplace=True)

df['TBG'].fillna(df['TBG'].mean(),inplace=True)
df.drop('TBG',axis=1,inplace=True)
df.drop('referral source',axis=1,inplace=True)
X = df.drop('binaryClass', axis=1)
df=df[df['binaryClass']==1]
'''
#determining only the most important columns,commenting to prevent from running everytime the model is loaded
x=df.iloc[:,0:27]
y=df.iloc[:,27]
model=LogisticRegression(max_iter=20000)

model.fit(x,y)
coeff=model.coef_[0]

logreg = LogisticRegression()
logreg.fit(X, y)
coefficients = logreg.coef_[0]
names = X.columns.tolist()

# Combine coefficients and variable names into a pandas DataFrame
df1 = pd.DataFrame({'name': names, 'coef': coefficients})

# Sort variables by absolute value of coefficient in descending order
df1 = df1.reindex(df1['coef'].abs().sort_values(ascending=False).index)

# Calculate the percentage contribution of each variable to the model
total_coef = np.abs(df1['coef']).sum()
df1['pct_contribution'] = np.abs(df1['coef']) / total_coef * 100

# Print the top 10 variables by percentage contribution
most_imp=np.array(df1.head(10))
req_col=most_imp[:,0]


#required columns are found out to be 
req_col=['on thyroxine', 'TSH measured', 'thyroid surgery', 'T4U', 'T3','T3 measured', 'tumor', 'goitre', 'query hypothyroid','TT4 measured']

'''
req_col=['on thyroxine', 'TSH measured', 'thyroid surgery', 'T4U', 'T3','T3 measured', 'tumor', 'goitre', 'query hypothyroid','TT4 measured','age']
df=df[req_col]

x=df.iloc[:,0:10].values
y=df.iloc[:,10]
y=y.astype(int)
model=RandomForestClassifier(n_estimators=100,random_state=0)
'''
finding the accuracy of the model ,commenting to prevent it from running everytime the model is loaded
x_tr,x_te,y_tr,y_te=train_test_split(x,y,test_size=0.2,random_state=0)
model=RandomForestClassifier(n_estimators=400,random_state=0)
model.fit(x_tr,y_tr)
y_pred=model.predict(x_te)



def accur(y_true, y_pred, margin=10):
    # Calculate the absolute difference between the true and predicted values
    diff = np.abs(y_true - y_pred)  
    # Count the number of values that differ by less than or equal to the margin
    num_correct = np.sum(diff <= margin)
    acc = num_correct / len(y_true)
    return acc


acc=accur(y_te,y_pred)
acc of 0.3314203730272597 with a margin of error of 10
'''
model.fit(x,y)

pickle.dump(model,open('age_thyroid.pkl','wb'))