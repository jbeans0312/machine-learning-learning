"""
CISC484: Machine Learning
Homework 3, Question 2
Authors: Disha Thakar, John Bean, Jenn Werth
"""

def map(mydata):
    newdata = []
    for instance in mydata:
        newdata.append([1, instance[0]*1.414, instance[1]*1.414, instance[0]*instance[1]*1.414, instance[0]*instance[0], instance[1]*instance[1]])
    return newdata


def predict(row, weights):

    activation = weights[0]

    for i in range(len(row) - 1):
        activation += weights[i + 1] * row[i]

    if activation >= 0:
        return 1
    else:
        return 0


# Estimate Perceptron weights using stochastic gradient descent
def ptalg(curr_data):
    curr_data = map(curr_data)
    epochs = 100
    eng = 0.1  # choose a suitably small learning rate value
    weights = [0.1 for i in range(len(curr_data[0]))]
    for epoch in range(epochs):
        sum_error = 0.0
        for row in curr_data:
            prediction = predict(row, weights)
            error = row[-1] - prediction
            sum_error += error ** 2
            weights[0] = weights[0] + eng * error
            for i in range(len(row) - 1):
                weights[i + 1] = weights[i + 1] + eng * error * row[i]
            if epoch%2 == 0:
            # print the predicted value on every other epoch 
                print('predicted value = ', prediction)
    return weights

def main():
    curr_data = [[0,0], [0,1], [1,0], [1,1]]
    weights = ptalg(curr_data)
    print(weights)

if __name__=="__main__":
    main()
