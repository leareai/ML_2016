# coding:utf-8
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt

import time


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


def batch_gradient_descent(input_x, output_y, base_rate, shrinking_rate, expending_rate, gradient_max=1):
    global w, loss, e
    min_gradient_value = 0.000001
    min_descent = 0.000001
    gradient_value = float('Inf')
    descent = float('Inf')

    count = 0
    e = output_y - input_x.dot(w)
    prev_loss = np.sqrt(e.dot(e) / e.index.size)
    while gradient_value > min_gradient_value and descent > min_descent:
        gradient = -2 * e.dot(input_x)
        gradient_value = np.sqrt(gradient.dot(gradient))
        rate = base_rate / gradient_value if (gradient_value > gradient_max) else base_rate
        w -= rate * gradient
        e = output_y - input_x.dot(w)
        loss = np.sqrt(e.dot(e) / e.index.size)
        descent = rate * gradient_value
        if prev_loss < loss:
            base_rate *= shrinking_rate
        else:
            base_rate *= expending_rate
        prev_loss = loss
        count += 1
        if count % 100 == 0:
            print 'grad:', gradient_value, 'rate:', rate, 'descent:', descent, ', loss:', loss, ', count:', count

    return w, loss


raw = load_data()
dataTrunk = slice_into_trunks(raw)
y, x, origX = get_data(dataTrunk)

w = np.ones(x.columns.size)
e = 0
loss = 0

if 'input_base_rate' not in globals():
    input_base_rate = 0.005

if 'shrk_rate' not in globals():
    shrk_rate = 0.8

if 'epd_rate' not in globals():
    epd_rate = 1.11

print 'base rate:', input_base_rate, 'shrinking rate:', shrk_rate, 'expending rate:', epd_rate
start_time = time.time()
w, loss = batch_gradient_descent(x, y, input_base_rate, shrk_rate, epd_rate)
print 'time:', (time.time() - start_time)