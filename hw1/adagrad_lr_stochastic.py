# coding:utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

import sys
from pandas import Series, DataFrame
from random import randint

import signal

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


def set_signal(log_func):
    def signal_handler(sig, frame):
        print '\n'
        log_func()
        print 'time:', (time.time() - start_time)
        if sig == signal.SIGQUIT:
            print'\nYou pressed Ctrl+\\'
            global loggable
            loggable = not loggable
        elif sig == signal.SIGINT:
            print'\nYou pressed Ctrl+C:'
            sys.exit(0)

    signal.signal(signal.SIGQUIT, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)


def stochastic_gradient_descent_with_adagrad(input_x, output_y, base_rate=0.1):
    global w, loss, e
    min_gradient_value = 1e-6
    min_descent = e-7
    rate = base_rate

    data_size = input_x.shape[0]
    feature_number = input_x.shape[1]
    descent = float('Inf')
    gradient_value = float('Inf')
    count = 0

    def log():
        global e, loss
        e = output_y - input_x.dot(w)
        loss = np.sqrt(e.dot(e) / data_size)
        print 'grad:', gradient_value, 'decent:', descent, ', loss:', loss, ', count:', count

    set_signal(log)

    rand_range = data_size - 1
    gradient_square_sum = np.zeros(feature_number)
    # while gradient_value > min_gradient_value or descent > min_descent:
    while gradient_value > min_gradient_value and descent > min_descent:
        index = randint(0, rand_range)
        y_picked = output_y[index]
        x_picked = input_x[index]
        e_picked = y_picked - x_picked.dot(w)
        gradient = -2 * e_picked * x_picked
        gradient_square_sum += gradient * gradient
        adagrad = np.sqrt(gradient_square_sum)
        adagrad_gradient = gradient / adagrad
        w -= rate * adagrad_gradient
        descent = np.sqrt(adagrad_gradient.dot(adagrad_gradient)) * rate
        gradient_value = np.sqrt(gradient.dot(gradient))
        if loggable and count % log_rate == 0:
            log()
        count += 1

    log()
    return w, loss


raw = load_data()
dataTrunk = slice_into_trunks(raw)
y, x, origX = get_data(dataTrunk)

w = np.ones(x.columns.size)
loss = e = 0

if 'loggable' not in globals():
    loggable = False

if 'log_rate' not in globals():
    log_rate = 100000

if 'adagrad_base_rate' not in globals():
    adagrad_base_rate = 0.1

print'Start iteration, you can pressed Ctrl+\\ to switch log, pressed Ctrl+C to force terminate'
print 'base rate:', adagrad_base_rate

start_time = time.time()
w, loss = stochastic_gradient_descent_with_adagrad(x.as_matrix(), y, adagrad_base_rate)
print 'time:', (time.time() - start_time)