import numpy as np

eng = 0.01
epoch = 0
training_set = [[1, -1, 0],
                [1, -2, 0],
                [1, -3, 0],
                [1, -4, 0],
                [1, 1, 1],
                [1, 2, 1],
                [1, 3, 1],
                [1, 4, 1]]
weights = [0.1, 0.1]
err_diff = 99999999
err_curr = 0
err_prev = 0


def compute_err(X, y, w):
    return np.sum((y-np.dot(X, w))**2)


while epoch < 20 and err_diff > 1:
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
    err_curr = compute_err(np.array(training_set)[:, :2], np.array(training_set)[:, -1], weights)
    err_diff = err_prev - err_curr
    err_diff = abs(err_diff)
    err_prev = err_curr
    epoch += 1
    print(err_diff)
