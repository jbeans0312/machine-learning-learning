from sklearn import svm
import pandas as pd
import numpy as np

# ignore that this says q2 dataset even though this is question 3
data = pd.read_csv('q2_datasetcsv.txt')
Y = data.iloc[:, 0]
X = data.iloc[:, 1:]

# train a model with the default C param
m1 = svm.SVC(C=1.0, kernel='linear')
m2 = svm.SVC(C=3.0, kernel='linear')

m1.fit(X, Y)
m2.fit(X, Y)

#add points to the training set
X.loc[len(X.index)] = [6, 8]
X.loc[len(X.index)] = [5, 3]

m1_prediction = m1.predict(X)
m2_prediction = m2.predict(X)

np.savetxt('./q3_predict1.txt', m1_prediction)
np.savetxt('./q3_predict2.txt', m2_prediction)
