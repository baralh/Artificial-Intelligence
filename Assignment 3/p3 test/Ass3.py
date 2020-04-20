#!/usr/bin/env python


#CMSC 409: Artificial Intelligence
# Project No. 3
# Due Oct. 29, 2019, noon
#Name: Heman Baral, Jedidiah Pottle, James Stallings




import numpy as np
import matplotlib.pyplot as plt
import random


class Energy_Consumption:
    def __init__(self, hrs, KW):
        self.hrs = hrs
        self.KW = KW


# splitting & loading the dataset in to consumption object
def data_load(file):
    test_objs = list()
    for i in range(16):
        data = file.readline().split(",")
        Consumption = Energy_Consumption(float(data[0]), float(data[1]))
        total_hrs = (Consumption.hrs - 5.00) / (20.00 - 5.00)
        Consumption.hrs = total_hrs
        test_objs.append(Consumption)
    return test_objs


def plot(test_list, polynomial, title):
    plt.xlabel('Hours')
    plt.ylabel('KW')
    plt.title('Prediction of Energy Consumption \n degree: ' + str(polynomial) + ' ' + title)
    x = list()
    for i in test_list[0]:
        x.append(i.hrs)
    x = np.array(x)

    for i in test_list[0]:
        plt.scatter(i.hrs, i.KW, c='b')
        if polynomial == 3:
            y = (current_weight[1] * x) + (current_weight[2]
                * (x ** 2)) + (current_weight[3]
                * (x ** 3)) + current_weight[0]
        elif polynomial == 2:
            y = (current_weight[1] * x) + \
                (current_weight[2] * (x ** 2)) + current_weight[0]
        elif polynomial == 1:
            y = (current_weight[1] * x) + current_weight[0]
        plt.plot(x, y, c='r')
    plt.show()



# Alpha value
a = 0.3
# Total number of iteration
total_iteration = 4000
current_weight = []
# train data size
data_size = 16

# Opening in read mode
data1 = open("Project3_data/train_data_1.txt", 'r')
data2 = open("Project3_data/train_data_2.txt", 'r')
data3 = open("Project3_data/train_data_3.txt", 'r')
data4 = open("Project3_data/test_data_4.txt", 'r')

set1 = data_load(data1)
set2 = data_load(data2)
set3 = data_load(data3)
set4 = data_load(data4)

train_data = [set1, set2, set3]


def create_model(values, total_iteration, data_size, a, polynomial = 1, x = ''):
    #polynomial = 1
    iteration = 0
    total_error_amount = 5
    #x = ''

    # randomizing weight
    for i in range(4):
        current_weight.append(round(random.uniform(-0.5, 0.5), 2))

    while (iteration < total_iteration and total_error_amount >= 5):
        iteration += 1
        total_total_error = 0
        for i in range(data_size):
            total_bias = 1 * current_weight[0]
            desired = values[i].KW
            net = 0.0

            # checking polynomial degree
            if polynomial == 1:
                net = (values[i].hrs * current_weight[1]) + total_bias
            elif polynomial == 2:
                net = (values[i].hrs * current_weight[1]) \
                      + ((values[i].hrs ** 2)
                        * current_weight[2]) + total_bias
            elif polynomial == 3:
                net = (values[i].hrs * current_weight[1]) \
                      + ((values[i].hrs ** 2) * current_weight[2]) + (
                            (values[i].hrs ** 3) * current_weight[3]) + total_bias

            total_error = desired - net
            total_total_error += total_error ** 2

            if (polynomial == 1):
                current_weight[0] += (a * total_error)
                current_weight[1] += (a * total_error) * values[i].hrs
            elif (polynomial == 2):
                current_weight[0] += (a * total_error)
                current_weight[1] += (a * total_error) * values[i].hrs
                current_weight[2] += (a * total_error) * (values[i].hrs ** 2)
            elif (polynomial == 3):
                current_weight[0] += (a * total_error)
                current_weight[1] += (a * total_error) * values[i].hrs
                current_weight[2] += (a * total_error) * (values[i].hrs ** 2)
                current_weight[3] += (a * total_error) * (values[i].hrs ** 3)

    print('polynomial Degree: ' + str(polynomial))
    print('total_error:', total_total_error)
    plot([values], polynomial, x)


# Testing day 4
for i in range(1, 4):
    current_weight.clear()
    for x in range(0, 3):
        create_model(train_data[x], total_iteration, data_size, a, i)
    create_model(set4, total_iteration, data_size, a, i, '(Day 4)')




