import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
test1 = pd.read_pickle('HW1_Q5/test_1.pkl')
train1 = pd.read_pickle('HW1_Q5/train_1.pkl')

feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI'
                , 'DiabetesPedigreeFunction', 'Age']

# divide data into features and outcome columns
test_X = test1[feature_cols]
test_y = test1.Outcome
train_X = train1[feature_cols]
train_y = train1.Outcome

# instantiate the model with default params
logreg1 = LogisticRegression(random_state=16, max_iter=900)

# fit the model with the data
logreg1.fit(train_X, train_y)

print(logreg1.coef_)

y_pred1 = logreg1.predict(test_X)
print(accuracy_score(test_y, y_pred1) *100)
print(y_pred1)






