import requests

import pandas as pd
import numpy as np
import sklearn

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics
dataset = pd.read_csv("dataset.csv")


dataset.dropna(inplace=True)
#Delete all features that are using strings
drop_dataset = dataset.drop(['URL', 'CHARSET', 'SERVER',
                       'WHOIS_COUNTRY', 'WHOIS_STATEPRO','WHOIS_REGDATE', 'WHOIS_UPDATED_DATE'], axis="columns")

new_dataset = drop_dataset;

X = new_dataset.drop(['Type'],axis = "columns")

print(new_dataset.shape);



y = new_dataset['Type']


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)



clf = svm.SVC(kernel='linear')
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test,y_pred))




