import numpy as np

epoch = 0
err = -1
eng = 0.01
training_set = [[1, 2, 1, 3, 6],
                [1, 5, 2, 7, 11],
                [1, 2, -2, 4, 0],
                [1, -3, 1, -2, 0],
                [1, -5, 3, -1, 1]]
weights = [0.1, 0.2, 0.1, 0.1]


def compute_err(X, y, w):
    return np.sum((y-np.dot(X, w))**2)


while epoch < 10 and err != 0:
    delta_w = [0, 0, 0, 0]
    # iterate over each weight

    for i in range(len(weights)):
        # iterate through each row in training set
        cum_sum = 0
        # calculate the dot product wx
        for k in range(len(training_set)):
            output = 0
            # calculate the dot product between training values and weights
            for j in range(len(weights)):
                output += weights[j]*training_set[k][j]
            # calculates the sum
            cum_sum += (training_set[k][-1] - output)*training_set[k][i]
        delta_w[i] = eng * cum_sum

    # recalculate the weights
    for l in range(len(weights)):
        weights[l] = weights[l] + delta_w[l]
    # recalculate weights array using delta_w array
    err = compute_err(np.array(training_set)[:, :4], np.array(training_set)[:,-1], weights)
    print(err)
    epoch += 1
