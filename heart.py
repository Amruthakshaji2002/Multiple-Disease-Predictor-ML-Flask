import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# loading the heart dataset to a pandas DataFrame
df=pd.read_csv(r'C:\Users\aswin.LAPTOP-5QCKUJE2\Downloads\Multiple-Disease-Predictor-ML-Flask-WebApp-main (1)\Multiple-Disease-Predictor-ML-Flask-WebApp-main\heart.csv')

# seperating the data and label into X and Y respectively
X=df.iloc[:,0:13]
Y=df['target']


X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, stratify=Y,random_state=1)

# Building the model using Logistic Regression
classifier = LogisticRegression()
classifier.fit(X_train, Y_train)


pickle.dump(classifier, open('heart.pkl', 'wb'))



