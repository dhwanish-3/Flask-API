import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

df = pd.read_csv("https://raw.githubusercontent.com/shellyannissa/Compute-Disease-Probability/main/survey%20lung%20cancer.csv")
encoder = LabelEncoder()
df['LUNG_CANCER']=encoder.fit_transform(df['LUNG_CANCER'])
df['GENDER']=encoder.fit_transform(df['GENDER'])
df=df[df['LUNG_CANCER']==1]
y=df['AGE']
a=df.columns.tolist()
a=['GENDER',
 'SMOKING',
 'YELLOW_FINGERS',
 'ANXIETY',
 'PEER_PRESSURE',
 'CHRONIC DISEASE',
 'FATIGUE ',
 'ALLERGY ',
 'WHEEZING',
 'ALCOHOL CONSUMING',
 'COUGHING',
 'SHORTNESS OF BREATH',
 'SWALLOWING DIFFICULTY',
 'CHEST PAIN',
 'LUNG_CANCER']
x=df[a]
model=RandomForestClassifier(n_estimators=200,random_state=0)
model.fit(x,y)

pickle.dump(model,open('age_lung_cancer.pkl','wb'))