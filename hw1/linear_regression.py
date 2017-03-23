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


raw = load_data()
dataTrunk = slice_into_trunks(raw)
y, x, origX = get_data(dataTrunk)

w = np.ones(x.columns.size)
base_rate = 0.005
gradient_value = 1
gradientMax = 1

count = 0
e = y - x.dot(w)
prev_loss = np.sqrt(e.dot(e) / e.index.size)
while gradient_value > 0.000001 and descent > 0.000001:
    e = y - x.dot(w)
    gradw = -2 * e.dot(x)
    gradient_value = np.sqrt(gradw.dot(gradw))
    rate = base_rate / gradient_value if (gradient_value > gradientMax) else base_rate
    w -= rate * gradw
    loss = np.sqrt(e.dot(e) / e.index.size)
    descent = rate * gradient_value
    if prev_loss < loss:
        base_rate *= 0.8
    else:
        base_rate *= 1.11
    prev_loss = loss
    count += 1
    if count % 100 == 0:
        print 'grad:', gradient_value, 'rate:', rate, 'descent:', descent, ', loss:', loss, ', count:', count
