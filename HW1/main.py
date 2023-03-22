epoch = 0
err = -1
eng = 0.01
training_set = [[1, 2, 1, 3, 6],
                [1, 5, 2, 7, 11],
                [1, 2, -2, 4, 0],
                [1, -3, 1, -2, 0],
                [1, -5, 3, -1, 1]]
weights = [0.1, 0.2, 0.1, 0.1]

while epoch <= 10 and err != 0:
    delta_w = [0, 0, 0, 0]
    #iterate through each row in the training set
    for i in range(4):
        output = 0
        #calculate the dot product wx
        for j in range(3):
            output += weights[j]*training_set[i][j]

        err = training_set[i][3] - output #calculate the error

        #update delta_w
        for k in range(3):
            delta_w[k] = delta_w[k] + eng * err * training_set[i][k]

    if err == 0:
        break
    for l in range(3):
        weights[l] = weights[l] + delta_w[l]
    #recalulate weights array using delta_w array
    print(err)
    epoch += 1
