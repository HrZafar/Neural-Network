import csv
import random
import math


def rand(x):
    if x == 0:
        return random.uniform(0, 1)
    else:
        return random.uniform(-1, 1)


def total_records():
    with open('NN-DATA.csv') as csvfile1:
        CSV = csv.reader(csvfile1, delimiter=',')
        firstline = True
        i = 0
        for line in CSV:
            if firstline:
                firstline = False
            else:
                i += 1
    return i


inp = [1, 1, 1]
bias = [[rand(0), rand(0)], [rand(0)]]
weights = [[[rand(1), rand(1), rand(1)], [rand(1), rand(1), rand(1)]], [[rand(1), rand(1)]]]
target = 0
learning_rate = 0
inputs = [[0, 0], [0]]
outputs = [[0, 0], [0]]
errors = [[1, 1], [1]]


def calc_inputs(layer):
    for i in range(len(inputs[layer])):
        w = 0
        for j in range(len(weights[layer][i])):
            if layer == 0:
                w = inp[j] * weights[layer][i][j] + w
            else:
                w = outputs[layer - 1][j] * weights[layer][i][j] + w
        inputs[layer][i] = w + bias[layer][i]


def calc_outputs(layer):
    for i in range(len(outputs[layer])):
        outputs[layer][i] = 1 / (1 + math.exp(-inputs[layer][i]))


def feed_forward():
    calc_inputs(0)
    calc_outputs(0)
    calc_inputs(1)
    calc_outputs(1)


def calc_errors(layer):
    if layer == 1:
        errors[layer][0] = outputs[layer][0] * (1 - outputs[layer][0]) * (target - outputs[layer][0])
    else:
        for i in range(len(errors[layer])):
            errors[layer][i] = outputs[layer][i] * (1 - outputs[layer][i]) * errors[layer + 1][0] * \
                               weights[layer + 1][0][i]


def update_bias(layer):
    for i in range(len(bias[layer])):
        bias[layer][i] = bias[layer][i] + errors[layer][i] * learning_rate


def update_weights(layer):
    if layer == 1:
        for i in range(len(weights[layer][0])):
            weights[layer][0][i] = weights[layer][0][i] + learning_rate * errors[layer][0] * outputs[layer - 1][i]
    else:
        for i in range(len(weights[layer])):
            for j in range(len(weights[layer][i])):
                weights[layer][i][j] = weights[layer][i][j] + learning_rate * errors[layer][i] * inp[j]


def back_propagation():
    calc_errors(1)
    calc_errors(0)
    update_bias(1)
    update_bias(0)
    update_weights(1)
    update_weights(0)


def min_max(column):
    with open('NN-DATA.csv') as csvfile1:
        CSV = csv.reader(csvfile1, delimiter=',')
        firstline = True
        data = []
        for line in CSV:
            if firstline:
                firstline = False
            else:
                data.append(float(line[column]))
        return min(data), max(data)


def normalize(x, min, max):
    return round((float(x) - min) / (max - min), 4)


op = []
op1 = 0
total = total_records()
check = False
cross_check = True
num = 0
test = False
while (check == False):
    print(num, num, num, num, num, num, num, num, num, num)
    op.clear()
    with open('NN-DATA.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        min_x1, max_x1 = min_max(0)
        min_x2, max_x2 = min_max(1)
        min_x3, max_x3 = min_max(2)
        min_y, max_y = min_max(3)
        firstline = True
        i = 0
        fitness = 0
        for row in readCSV:
            errors = [[1, 1], [1]]
            learning_rate = 1 / (i + 1)
            if firstline:
                firstline = False
            else:
                inp[0] = normalize(row[0], min_x1, max_x1)
                inp[1] = normalize(row[1], min_x2, max_x2)
                inp[2] = normalize(row[2], min_x3, max_x3)
                target = normalize(row[3], min_y, max_y)
                # print(inp)
                # print(target)
                j = 0
                while ((errors[1][0] > 0.001 or errors[1][0] < -0.001) and test == False):
                    feed_forward()
                    back_propagation()
                    # print('################################LEARNING#######################################')
                    # print('i=', i, '  j=', j)
                    # print('INP', inp)
                    # print('TARGET', target)
                    # print('INPUTS', inputs)
                    # print('OUTPUTS', outputs)
                    # print('ERRORS', errors)
                    # print('BIAS', bias)
                    # print('WEIGHTS', weights)
                    j += 1
                if test == True:
                    feed_forward()
                    calc_errors(1)
                    op.append(round(outputs[1][0], 4))
                    print('#############################TESTING###################################')
                    print('i=', i, '  j=', j)
                    print('INP', inp)
                    print('TARGET', target)
                    print('INPUTS', inputs)
                    print('OUTPUTS', outputs)
                    print('ERRORS', errors)
                    print('BIAS', bias)
                    print('WEIGHTS', weights)
                    j += 1
                    if -0.005 < errors[1][0] and errors[1][0] < 0.005:
                        fitness += 1
                if i == total:
                    cross_check = False
                    print(len(op), op)
                    if op1 < fitness:
                        op1 = fitness
                    print('fitness', fitness, op1)
                    if fitness >= 500 and test == True:
                        cross_check = True
                        print('hjhjfdfddgh')
                    if test == False:
                        test = True
                    elif test == True:
                        test = False
            i += 1
    if cross_check != False:
        check = True
    num += 1
print('end')
print(len(op), op)
