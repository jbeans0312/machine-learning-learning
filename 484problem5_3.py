import pandas as pd
pd.options.display.max_columns = 999
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
test3 = pd.read_pickle('HW1_Q5/test_3.pkl')
train3 = pd.read_pickle('HW1_Q5/train_3.pkl')
feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI'
                , 'DiabetesPedigreeFunction', 'Age']

# divide data into features and outcome columns
test_X = test3[feature_cols]
test_y = test3.Outcome
train_X = train3[feature_cols]
train_y = train3.Outcome

# instantiate the model with default params
logreg3 = LogisticRegression(random_state=16, max_iter=900)

logreg3.fit(train_X, train_y)

print(logreg3.coef_)

y_pred3 = logreg3.predict(test_X)
print(accuracy_score(test_y, y_pred3) * 100)
print(y_pred3)