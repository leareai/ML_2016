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


raw = load_data()
dataTrunk = slice_into_trunks(raw)
y, x, origX = get_data(dataTrunk)

length = x.columns.size
w = np.ones(length)
base_rate = 0.1
rate = base_rate
gradient_value = 1
gradientMax = 1
count = 0
size = x.index.size
gradientSquareSum = np.zeros(length)
descent = np.sqrt(w.dot(w))
gradw_value=1
while gradw_value > 0.00001 or descent > 0.0000001:
    index = randint(0, size - 1)
    y_head = y[index]
    x_head = x.ix[index]
    e_head = y_head - x_head.dot(w)
    gradw = -2 * e_head * x_head
    gradw_sqr = gradw * gradw
    gradientSquareSum += gradw_sqr
    adagrad = np.sqrt(gradientSquareSum)
    adagrad_gradw = gradw / adagrad
    w -= rate * adagrad_gradw
    if count % 1000 == 0:
        descent = np.sqrt(adagrad_gradw.dot(adagrad_gradw)) * rate
        gradient_value = np.sqrt(gradw.dot(gradw))
        e = y - x.dot(w)
        loss = np.sqrt(e.dot(e) / size)
        print 'grad:', gradient_value, 'decent:', descent, ', loss:', loss, ', count:', count
    count += 1
