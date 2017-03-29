# coding:utf-8
import numpy as np
import pandas as pd
import signal

import sys
from pandas import Series, DataFrame
import matplotlib.pyplot as plt

import time


def load_data(selected_items):
    raw_data = pd.read_csv('./data/train.csv', na_values='NR', encoding='big5')
    # raw_data.ix[raw_data[u'測項'] == 'RAINFALL', 3:] = raw_data.ix[raw_data[u'測項'] == 'RAINFALL', 3:].fillna(0)
    raw_data.fillna(0, inplace='True')
    raw_output = raw_data.ix[raw_data[u'測項'] == 'PM2.5', 3:]
    raw_data = raw_data.ix[raw_data[u'測項'].isin(selected_items)]
    raw_data.drop(raw_data.columns[0:3], axis=1, inplace=True)
    return raw_data, raw_output


def slice_into_trunks(raw_data, row_len):
    """split data by month"""
    trunk = []
    row_offset = 18
    row_index = 0
    indexes = range(row_len)
    for mon in range(12):
        month_trunk = []
        for day in range(20):
            data_per_day = raw_data.ix[row_index:row_index + row_offset - 1]
            data_per_day.index = indexes
            month_trunk.append(data_per_day)
            row_index += row_offset
        trunk.append(pd.concat(month_trunk, axis=1))
    return trunk


def get_output(trunk, hours):
    temp_array = np.array([data.ix[0, hours:].values for data in trunk])
    temp_shape = temp_array.shape
    return temp_array.reshape(temp_shape[0]*temp_shape[1], )


def get_data(d_trunk):
    i = 0
    hours = 9
    data_dict = {}
    for data in d_trunk:
        for j in range(data.columns.size - hours):
            # vector = data.ix[:, j:j + hours].values.T.flatten()
            vector = data.ix[:, j:j + hours].values.flatten()
            data_dict[i] = vector
            i += 1

    input_data = DataFrame(data_dict).T
    normal_data = (input_data - input_data.mean()) / input_data.std()
    normal_data['b'] = 1
    input_data['b'] = 1
    return normal_data, input_data


def set_signal(log_func):
    def signal_handler(sig, frame):
        print '\n'
        log_func()
        print 'time:', (time.time() - start_time)
        if sig == signal.SIGQUIT:
            print'You pressed Ctrl+\\'
            global loggable
            loggable = not loggable
        elif sig == signal.SIGINT:
            print'You pressed Ctrl+C'

            sys.exit(0)

    signal.signal(signal.SIGQUIT, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)


def batch_gradient_descent(input_x, output_y, base_rate, shrinking_rate, expending_rate, gradient_max=1):
    global w, loss, e
    # min_gradient_value = 1
    # min_descent = 1e-5
    min_gradient_value = 1e-6
    min_descent = 1e-15
    gradient_value = float('Inf')
    descent = float('Inf')
    rate = 0
    count = 0
    data_size = input_x.shape[0]

    def log():
        print 'grad:', gradient_value, 'rate:', rate, 'descent:', descent, ', loss:', loss, ', count:', count

    set_signal(log)

    e = output_y - input_x.dot(w)

    prev_loss = np.sqrt(e.dot(e) / data_size)
    while gradient_value > min_gradient_value or descent > min_descent:
        gradient = -2 * e.dot(input_x)
        gradient_value = np.sqrt(gradient.dot(gradient))
        rate = base_rate / gradient_value if (gradient_value > gradient_max) else base_rate
        w -= rate * gradient
        e = output_y - input_x.dot(w)
        loss = np.sqrt(e.dot(e) / data_size)
        descent = rate * gradient_value
        if prev_loss < loss:
            base_rate *= shrinking_rate
        else:
            base_rate *= expending_rate
        prev_loss = loss
        if loggable and count % log_rate == 0:
            log()
        count += 1

    log()
    return w, loss


data_item = {
 # u'AMB_TEMP',
 # u'CH4',
 # u'CO',
 # u'NMHC',
 # u'NO',
 # u'NO2',
 # u'NOx',
 # u'O3',
 # u'PM10',
 u'PM2.5',
 u'RAINFALL',
 u'RH',
 u'SO2',
 u'THC',
 u'WD_HR',
 u'WIND_DIREC',
 u'WIND_SPEED',
 u'WS_HR'
}

raw, raw_y = load_data(data_item)
dataTrunk = slice_into_trunks(raw, len(data_item))
y_trunk = slice_into_trunks(raw_y, 1)
y = get_output(y_trunk, 9)
x, origX = get_data(dataTrunk)

e = 0
loss = 0

if 'w' not in globals():
    w = np.ones(x.columns.size)

if 'loggable' not in globals():
    loggable = False

if 'log_rate' not in globals():
    log_rate = 1200

if 'input_base_rate' not in globals():
    input_base_rate = 0.005

if 'shrk_rate' not in globals():
    shrk_rate = 0.8

if 'epd_rate' not in globals():
    epd_rate = 1.11

if 'power' not in globals():
    power = 1

print'Start iteration, you can pressed Ctrl+\\ to switch log, pressed Ctrl+C to force terminate'
print 'base rate:', input_base_rate, 'shrinking rate:', shrk_rate, 'expending rate:', epd_rate
start_time = time.time()
w, loss = batch_gradient_descent(x.as_matrix(), y, input_base_rate, shrk_rate, epd_rate)
print 'time:', (time.time() - start_time)