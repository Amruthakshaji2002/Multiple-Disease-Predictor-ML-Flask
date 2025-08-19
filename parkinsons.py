import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
import pickle

# loading the Parkinsons Disease dataset to a pandas DataFrame
df=pd.read_csv(r'C:\Users\aswin.LAPTOP-5QCKUJE2\Downloads\Multiple-Disease-Predictor-ML-Flask-WebApp-main (1)\Multiple-Disease-Predictor-ML-Flask-WebApp-main\ParkinsonsDisease.csv')

# drop name column
df=df.drop(columns='name', axis=1)
df.head()

# seperating the data and label into X and Y respectively
X=df.drop(columns='status',axis=1)
Y=df['status']


X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, stratify=Y,random_state=2)

# Building the model using Support Vector Machine classifier
classifier = svm.SVC(kernel='linear')

# training the SVM classifier
classifier.fit(X_train, Y_train)


pickle.dump(classifier, open('parkinsons.pkl', 'wb'))
  

