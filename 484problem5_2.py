import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
test2 = pd.read_pickle('HW1_Q5/test_2.pkl')
train2 = pd.read_pickle('HW1_Q5/train_2.pkl')

feature_cols = ['Pregnancies', 'X', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI'
                , 'DiabetesPedigreeFunction', 'Age']

test_X = test2[feature_cols]
test_y = test2.Outcome
train_X = train2[feature_cols]
train_y = train2.Outcome

logreg2 = LogisticRegression(random_state=16, max_iter=900)

logreg2.fit(train_X, train_y)

print(logreg2.coef_)

y_pred2 = logreg2.predict(test_X)
print(accuracy_score(test_y, y_pred2) *100)
print(y_pred2)
