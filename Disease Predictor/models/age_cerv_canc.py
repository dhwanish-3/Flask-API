import pandas as pd
import numpy as np
import pickle

df=pd.read_csv("https://raw.githubusercontent.com/shellyannissa/Compute-Disease-Probability/main/kag_risk_factors_cervical_cancer.csv")
df = df.replace('?', np.NaN)
df=df.drop(['STDs: Time since first diagnosis', 'STDs: Time since last diagnosis'],axis=1)
df = df.dropna(axis=1, how='all')
df['Number of sexual partners']=pd.to_numeric(df['Number of sexual partners'],errors='coerce')
df['Number of sexual partners'].fillna(df['Number of sexual partners'].mean(),inplace=True)
df['First sexual intercourse']=pd.to_numeric(df['First sexual intercourse'],errors='coerce')
df['First sexual intercourse'].fillna(df['First sexual intercourse'].mean(),inplace=True)
df['Num of pregnancies']=pd.to_numeric(df['Num of pregnancies'],errors='coerce')
df['Num of pregnancies'].fillna(df['Num of pregnancies'].mean(),inplace=True)
df['Smokes']=pd.to_numeric(df['Smokes'],errors='coerce')
df['Smokes'].fillna(df['Smokes'].mean(),inplace=True)
df['Smokes (years)']=pd.to_numeric(df['Smokes (years)'],errors='coerce')
df['Smokes (years)'].fillna(df['Smokes (years)'].mean(),inplace=True)
df['Smokes (packs/year)']=pd.to_numeric(df['Smokes (packs/year)'],errors='coerce')
df['Smokes (packs/year)'].fillna(df['Smokes (packs/year)'].mean(),inplace=True)
df['Hormonal Contraceptives']=pd.to_numeric(df['Hormonal Contraceptives'],errors='coerce')
df['Hormonal Contraceptives'].fillna(df['Hormonal Contraceptives'].mean(),inplace=True)
df['Hormonal Contraceptives (years)']=pd.to_numeric(df['Hormonal Contraceptives (years)'],errors='coerce')
df['Hormonal Contraceptives (years)'].fillna(df['Hormonal Contraceptives (years)'].mean(),inplace=True)
df['IUD']=pd.to_numeric(df['IUD'],errors='coerce')
df['IUD'].fillna(df['IUD'].mean(),inplace=True)
df['STDs']=pd.to_numeric(df['STDs'],errors='coerce')
df['STDs'].fillna(df['STDs'].mean(),inplace=True)
df['IUD (years)']=pd.to_numeric(df['IUD (years)'],errors='coerce')
df['IUD (years)'].fillna(df['IUD (years)'].mean(),inplace=True)
df['STDs (number)']=pd.to_numeric(df['STDs (number)'],errors='coerce')
df['STDs (number)'].fillna(df['STDs (number)'].mean(),inplace=True)
df['STDs:condylomatosis']=pd.to_numeric(df['STDs:condylomatosis'],errors='coerce')
df['STDs:condylomatosis'].fillna(df['STDs:condylomatosis'].mean(),inplace=True)
df['STDs:cervical condylomatosis']=pd.to_numeric(df['STDs:cervical condylomatosis'],errors='coerce')
df['STDs:cervical condylomatosis'].fillna(df['STDs:cervical condylomatosis'].mean(),inplace=True)
df['STDs:vaginal condylomatosis']=pd.to_numeric(df['STDs:vaginal condylomatosis'],errors='coerce')
df['STDs:vaginal condylomatosis'].fillna(df['STDs:vaginal condylomatosis'].mean(),inplace=True)
df['STDs:vulvo-perineal condylomatosis']=pd.to_numeric(df['STDs:vulvo-perineal condylomatosis'],errors='coerce')
df['STDs:vulvo-perineal condylomatosis'].fillna(df['STDs:vulvo-perineal condylomatosis'].mean(),inplace=True)
df['STDs:syphilis']=pd.to_numeric(df['STDs:syphilis'],errors='coerce')
df['STDs:syphilis'].fillna(df['STDs:syphilis'].mean(),inplace=True)
df['STDs:pelvic inflammatory disease']=pd.to_numeric(df['STDs:pelvic inflammatory disease'],errors='coerce')
df['STDs:pelvic inflammatory disease'].fillna(df['STDs:pelvic inflammatory disease'].mean(),inplace=True)
df['STDs:genital herpes']=pd.to_numeric(df['STDs:genital herpes'],errors='coerce')
df['STDs:genital herpes'].fillna(df['STDs:genital herpes'].mean(),inplace=True)
df['STDs:molluscum contagiosum']=pd.to_numeric(df['STDs:molluscum contagiosum'],errors='coerce')
df['STDs:molluscum contagiosum'].fillna(df['STDs:molluscum contagiosum'].mean(),inplace=True)
df['STDs:AIDS']=pd.to_numeric(df['STDs:AIDS'],errors='coerce')
df['STDs:AIDS'].fillna(df['STDs:AIDS'].mean(),inplace=True)
df['STDs:HIV']=pd.to_numeric(df['STDs:HIV'],errors='coerce')
df['STDs:HIV'].fillna(df['STDs:HIV'].mean(),inplace=True)
df['STDs:Hepatitis B']=pd.to_numeric(df['STDs:Hepatitis B'],errors='coerce')
df['STDs:Hepatitis B'].fillna(df['STDs:Hepatitis B'].mean(),inplace=True)
df['STDs:HPV']=pd.to_numeric(df['STDs:HPV'],errors='coerce')
df['STDs:HPV'].fillna(df['STDs:HPV'].mean(),inplace=True)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
y=df['Age']
'''
x=df.iloc[:,0:26]
y=df.iloc[:,26]
model=LogisticRegression(max_iter=20000)
# model=RandomForestClassifier()
model.fit(x,y)
coeff=model.coef_[0]

logreg = LogisticRegression()
logreg.fit(x, y)
coefficients = logreg.coef_[0]
names = x.columns.tolist()

# Combine coefficients and variable names into a pandas DataFrame
df1 = pd.DataFrame({'name': names, 'coef': coefficients})

# Sort variables by absolute value of coefficient in descending order
df1 = df1.reindex(df1['coef'].abs().sort_values(ascending=False).index)

# Calculate the percentage contribution of each variable to the model
total_coef = np.abs(df1['coef']).sum()
df1['pct_contribution'] = np.abs(df1['coef']) / total_coef * 100
df1['pct_contribution']
it is determined that most important columns are ['STDs:HPV','STDs:condylomatosis','STDs:vulvo-perineal condylomatosis','Smokes','IUD','STDs:HIV','STDs:syphilis','STDs: Number of diagnosis','STDs (number)']
'''
req_col=['STDs:HPV','STDs:condylomatosis','STDs:vulvo-perineal condylomatosis','Smokes','IUD','STDs:HIV','STDs:syphilis','STDs: Number of diagnosis','STDs (number)']
df=df[req_col]
x=df.iloc[:,:]
model=RandomForestClassifier(n_estimators=200,random_state=0)
model.fit(x,y)

pickle.dump(model,open('age_cerv_canc.pkl','wb'))