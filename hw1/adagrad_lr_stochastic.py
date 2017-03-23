# coding:utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from pandas import Series, DataFrame
from random import randint


def load_data():
    raw_data = pd.read_csv('./data/train.csv', na_values='NR', encoding='big5')
    raw_data.ix[raw_data[u'測項'] == 'RAINFALL', 3:] = raw_data.ix[raw_data[u'測項'] == 'RAINFALL', 3:].fillna(0)
    raw_data.drop(raw_data.columns[0:3], axis=1, inplace=True)
    return raw_data


def slice_into_trunks(raw_data):
    """split data by month"""
    trunk = []
    row_offset = 18
    row_index = 0
    indexes = range(row_offset)
    for mon in range(12):
        month_trunk = []
        for day in range(20):
            data_per_day = raw_data.ix[row_index:row_index + row_offset - 1]
            data_per_day.index = indexes
            month_trunk.append(data_per_day)
            row_index += row_offset
        trunk.append(pd.concat(month_trunk, axis=1))
    return trunk


def get_data(d_trunk):
    i = 0
    hours = 9
    data_dict = {}
    output = []
    output_index = -9
    for data in d_trunk:
        for j in range(data.columns.size - hours + 1):
            vector = data.ix[:, j:j + hours].values.T.flatten()
            data_dict[i] = vector
            output.append(vector[output_index])
            i += 1

    output.pop(0)
    del data_dict[i - 1]
    input_data = DataFrame(data_dict).T
    normal_data = (input_data - input_data.mean()) / input_data.std()
    normal_data['b'] = 1
    input_data['b'] = 1
    return output, normal_data, input_data


def stochastic_gradient_descent_with_adagrad(input_x, output_y, base_rate=0.1):
    global w, loss, e
    min_gradient_value = 0.000001
    min_descent = 0.0000001
    length = input_x.columns.size
    rate = base_rate
    count = 0
    size = input_x.index.size
    gradient_square_sum = np.zeros(length)

    descent = float('Inf')
    gradw_value = float('Inf')
    while gradw_value > min_gradient_value or descent > min_descent:
        index = randint(0, size - 1)
        y_picked = output_y[index]
        x_picked = input_x.ix[index]
        e_picked = y_picked - x_picked.dot(w)
        gradient = -2 * e_picked * x_picked
        gradient_square_sum += gradient * gradient
        adagrad = np.sqrt(gradient_square_sum)
        adagrad_gradient = gradient / adagrad
        w -= rate * adagrad_gradient
        descent = np.sqrt(adagrad_gradient.dot(adagrad_gradient)) * rate
        if count % 1000 == 0:
            gradient_value = np.sqrt(gradient.dot(gradient))
            e = output_y - input_x.dot(w)
            loss = np.sqrt(e.dot(e) / size)
            print 'grad:', gradient_value, 'decent:', descent, ', loss:', loss, ', count:', count
        count += 1

    return w, loss

raw = load_data()
dataTrunk = slice_into_trunks(raw)
y, x, origX = get_data(dataTrunk)

w = np.ones(x.columns.size)
loss = e = 0

if 'adagrad_base_rate' not in globals():
    adagrad_base_rate = 0.1

print 'base rate:', adagrad_base_rate
start_time = time.time()
w, loss = stochastic_gradient_descent_with_adagrad(x, y, adagrad_base_rate)
print 'time:', (time.time() - start_time)