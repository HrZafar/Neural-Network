import random
import csv
import math


def insertionSort(alist):
    for i in range(1, len(alist)):
        key = alist[i]
        j = i - 1
        while j >= 0 and key['fitness'] > alist[j]['fitness']:
            alist[j + 1] = alist[j]
            j -= 1
        alist[j + 1] = key
    return alist


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


def calc_inputs(layer, inp, weights, bias, inputs, outputs):
    for i in range(len(inputs[layer])):
        w = 0
        for j in range(len(weights[layer][i])):
            if layer == 0:
                w = inp[j] * weights[layer][i][j] + w
            else:
                w = outputs[layer - 1][j] * weights[layer][i][j] + w
        inputs[layer][i] = w + bias[layer][i]
    return inputs


def calc_outputs(layer, inputs, outputs):
    for i in range(len(outputs[layer])):
        outputs[layer][i] = 1 / (1 + math.exp(-inputs[layer][i]))
    return outputs


def feed_forward(i, w, b, ip, op):
    ip = calc_inputs(0, i, w, b, ip, op)
    op = calc_outputs(0, ip, op)
    ip = calc_inputs(1, i, w, b, ip, op)
    op = calc_outputs(1, ip, op)
    return op[1][0]


def calc_errors(output, target):
    error = output * (1 - output) * (target - output)
    return error


def fitness(list):
    fitness = 0
    inp = [1, 1, 1]
    weights = [[[list[0], list[1], list[2]], [list[3], list[4], list[5]]], [[list[6], list[7]]]]
    bias = [[list[8], list[9]], [list[10]]]
    inputs = [[0, 0], [0]]
    outputs = [[0, 0], [0]]
    op.clear()
    tr.clear()
    with open('NN-DATA.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        min_x1, max_x1 = min_max(0)
        min_x2, max_x2 = min_max(1)
        min_x3, max_x3 = min_max(2)
        min_y, max_y = min_max(3)
        firstline = True
        i = 0
        for row in readCSV:
            if firstline:
                firstline = False
            else:
                i += 1
                inp[0] = normalize(row[0], min_x1, max_x1)
                inp[1] = normalize(row[1], min_x2, max_x2)
                inp[2] = normalize(row[2], min_x3, max_x3)
                target = normalize(row[3], min_y, max_y)
                output = feed_forward(inp, weights, bias, inputs, outputs)
                op.append(output)
                tr.append(target)
                error = calc_errors(output, target)
                if -0.005 <= error <= 0.005:
                    fitness += 1
    return fitness


def gen_individuals(n, b):
    list = []
    for i in range(b):
        dict = {}
        list1 = []
        for i in range(n):
            list1.append(random.uniform(-1, 1))
        dict['NN'] = list1
        dict['fitness'] = fitness(dict['NN'])
        list.append(dict)
    return list


def select_networks(n):
    p1 = random.randint(0, n - 1)
    p2 = random.randint(0, n - 1)
    while p1 == p2:
        p2 = random.randint(0, n - 1)
    return p1, p2


def mutation(child):
    probability = random.randint(1, 100)
    if probability > 25:
        mutation_value = random.randint(1, 100)
        if mutation_value > 50:
            mutation_value = .02
        else:
            mutation_value = -.02
        a, b = select_networks(len(child))
        temp = child[a] + mutation_value
        temp1 = child[b] + mutation_value
        if -1 <= temp <= 1:
            child[a] = temp
        if -1 <= temp1 <= 1:
            child[b] = temp1
    return child


def crossover(list, p1, p2):
    ch = []
    child = {}
    for i in range(round(len(list[p1]['NN']) / 2)):
        ch.append(list[p1]['NN'][i])
    j = i + 1
    while i < len(list[p2]['NN']) - 1:
        k = 0
        while k < len(ch):
            if list[p2]['NN'][j] == ch[k]:
                j += 1
                k = 0
                if j == len(list[p2]['NN']):
                    j = 0
            else:
                k += 1
        if k == len(ch):
            ch.append(list[p2]['NN'][j])
            i += 1
    ch = mutation(ch)
    child['NN'] = ch
    child['fitness'] = fitness(ch)
    return child


def gen_childs(list, p1, p2):
    return crossover(list, p1, p2), crossover(list, p2, p1)


op = []
tr = []
check = False
generations = 0
network = []
n = 11
Neural_networks = 25
child_networks = 20
childs = []
generation = gen_individuals(n, Neural_networks)
i = 0
while check == False and i < len(generation):
    if generation[i]['fitness'] >= 500:
        check = True
        network = generation[i]
    i += 1
while check == False:
    for j in range(int(child_networks / 2)):
        NN1, NN2 = select_networks(Neural_networks)
        childs.extend(gen_childs(generation, NN1, NN2))
    generation_1 = generation + childs
    generation_1 = insertionSort(generation_1)
    for k in range(len(generation)):
        generation[k] = generation_1[k]
        if generation[k]['fitness'] >= 100:
            check = True
            network = generation[k]
    print(generations, generation_1[0]['fitness'])
    generations += 1
print(generations)
print(network)
print(op)
print(tr)