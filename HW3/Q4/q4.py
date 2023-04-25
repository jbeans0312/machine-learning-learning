from sklearn import svm
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np


# load data

training = pd.read_csv('q3_training.txt', sep=" ", header=None)
train_Y = training.iloc[:, 0]
train_X = training.iloc[:, 1:]

testing = pd.read_csv('q3_test.txt', sep=" ", header=None)
test_Y = testing.iloc[:, 0]
test_X = testing.iloc[:, 1:]

"""We are interested in observing the number
of mis-classifications with linear, quadratic and rbf
kernels. For quadratic kernel choose poly and set degree
to two."""

m_lin = svm.SVC(C=1.0, kernel='linear')
m_quad = svm.SVC(C=1.0, kernel='poly', degree=2)

# LINEAR SVM
m_lin.fit(train_X, train_Y)
m_lin_pred = m_lin.predict(test_X)
accuracy_lin = accuracy_score(test_Y, m_lin_pred)
np.savetxt('./q4_predict_lin.txt', m_lin_pred)
print('Linear accuracy: ', accuracy_lin)

# QUADRATIC SVM
m_quad.fit(train_X, train_Y)
m_quad_pred = m_quad.predict(test_X)
accuracy_quad = accuracy_score(test_Y, m_quad_pred)
np.savetxt('./q4_predict_quad.txt', m_quad_pred)
print('\nQuadratic accuracy: ', accuracy_quad)

print('\nTesting C values for RBF kernel')
print('-------------------------------')
# RBF SVM
for x in range(5):
    c = pow(10, x)

    m_rbf = svm.SVC(C=c, kernel='rbf')
    m_rbf.fit(train_X, train_Y)
    m_rbf_pred = m_rbf.predict(test_X)
    accuracy_rbf = accuracy_score(test_Y, m_rbf_pred)
    print("For C value: ", c, " the accuracy is: ", accuracy_rbf)

print('\nAfter brute-force testing values between 100-10000')
print('we notice that the accuracy never exceeds 99.75%,\n'
      'so we take 1000 as the C value')

m_rbf = svm.SVC(C=1000, kernel='rbf')
m_rbf.fit(train_X, train_Y)
m_rbf_pred = m_rbf.predict(test_X)
np.savetxt('./q4_predict_rbf.txt', m_rbf_pred)


