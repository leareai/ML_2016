# coding:utf-8
import numpy as np
import pandas as pd
import signal

import sys
import scipy.sparse.linalg
from pandas import Series, DataFrame
import matplotlib.pyplot as plt

import time


def load_data(selected_items):
    raw_data = pd.read_csv('./data/train.csv', na_values='NR', encoding='big5')
    # raw_data.ix[raw_data[u'測項'] == 'RAINFALL', 3:] = raw_data.ix[raw_data[u'測項'] == 'RAINFALL', 3:].fillna(0)
    raw_data.fillna(0, inplace='True')
    raw_output = raw_data.ix[raw_data[u'測項'] == 'PM2.5', 3:]
    raw_data = raw_data.ix[raw_data[u'測項'].isin(selected_items)]

    # Dump negative value
    # aa = raw_data.drop(raw_data.columns[0:3], axis=1)
    # a_set = set()
    # for i in aa:
    #     print(i, aa[aa[i] < 0].index.tolist())
    #     a_set.update(aa[aa[i] < 0].index.tolist())
    # for i in sorted(list(a_set)):
    #     print (raw_data.ix[i, :])

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


def get_output(trunk, hrs):
    temp_array = np.array([data.ix[0, hrs:].values for data in trunk])
    temp_shape = temp_array.shape
    return temp_array.reshape(temp_shape[0]*temp_shape[1], )


def fix_value(feature, index):
    # print('fix>>', feature[index], index)
    last_index = len(feature) - 1
    if 0 < index < last_index:
        left = index-1
        right = index+1
        if feature[left] > 0 and feature[right] > 0:
            feature[index] = (feature[left] + feature[right])/2
    # print('fix<<', feature[index])


def get_data(d_trunk, start_hr, end_hr, output):
    global b_normal
    i = 0
    data_dict = {}
    for data in d_trunk:
        data_array = data.values
        data_len = data_array.shape[1]
        # interpolate if only one missing data
        [fix_value(yy, count, ) for yy in data_array for count in range(data_len) if yy[count] < 0]
        for j in range(data.columns.size - end_hr):
            vector = data_array[:, j+start_hr:j + end_hr]
            vector = vector[:, start_hr:]
            vector = vector.flatten()
            data_dict[i] = vector
            i += 1

    input_data = DataFrame(data_dict).T
    output = DataFrame(output)
    output.index = input_data.index

# filter out negative value
    a_list = []
    for idx in input_data.index:
        if np.any(input_data.ix[idx] < 0):
            a_list.append(idx)
    input_data = input_data.drop(a_list)
    output = output.drop(a_list)

    output = output.as_matrix();
    output = output.reshape(output.shape[0],);

    if b_normal:
        normal_data = (input_data - input_data.mean()) / input_data.std()
    else:
        normal_data = input_data.copy()

    normal_data_base = normal_data
    normal_data_power = normal_data
    for j in range(1, power_value):
        normal_data_power = normal_data_power * normal_data_base
        normal_data = pd.concat([normal_data, normal_data_power], axis=1, ignore_index=True)

    normal_data['b'] = 1
    return normal_data, input_data, output


def get_opt(reg):
    global atol, btol, conlim, maxiter, show
    atol = 1e-16; btol=1e-16; conlim=1e8; maxiter=100000; show=False
    ret = scipy.sparse.linalg.lsmr(x.as_matrix(), y, np.sqrt(reg), atol, btol, conlim, maxiter, show)
    print('istop:', ret[1], 'iter:', ret[2])
    ret = ret[0]
    ee = (y-x.dot(ret)).dot(y-x.dot(ret))
    print('reg:', reg, 'loss:', (np.sqrt((ee + reg*ret.dot(ret))/x.shape[0]), np.sqrt(ee/x.shape[0])))
    return ret


def get_model(in_x, in_y, reg):
    n_col = in_x.shape[1]
    w1 = np.linalg.lstsq(in_x.T.dot(in_x) + reg * np.identity(n_col), in_x.T.dot(in_y))
    w1 = w1[0]
    ee = (in_y - in_x.dot(w1)).dot(in_y - in_x.dot(w1))
    print('reg:', reg, 'loss:', (np.sqrt((ee + reg * w1.dot(w1)) / in_x.shape[0]), np.sqrt(ee / in_x.shape[0])))
    return w1


def least_square(in_x, in_y):
    ret = np.linalg.lstsq(in_x, in_y)
    print('least square loss: ', np.sqrt(ret[1] / len(in_y)))
    return ret[0]


test_item = {
 u'AMB_TEMP',
 # u'CH4',
 # u'CO',
 u'NMHC',
 u'NO',
 # u'NO2',
 # u'NOx',
 u'O3',
 u'PM10',
 u'PM2.5',
 u'RAINFALL',
 # u'RH',
 # u'SO2',
 # u'THC',
 u'WD_HR',
 u'WIND_DIREC',
 u'WIND_SPEED',
 u'WS_HR'
}


if 'power_value' not in globals():
    power_value = 2

if 'start_hour' not in globals():
    start_hour = 0

if 'b_normal' not in globals():
    b_normal = True;

if 'regulation' not in globals():
    regulation = 57 # best private
    # regulation = 3 # best public with good private

end_hour = 9


print('power_value', power_value, 'b_normal:', b_normal, 'start_hour:', start_hour, 'regulation:', regulation)
print(test_item)

raw, raw_y = load_data(test_item)
data_trunk = slice_into_trunks(raw, len(test_item))
y_trunk = slice_into_trunks(raw_y, 1)
y = get_output(y_trunk, end_hour)
x, origX, y = get_data(data_trunk, start_hour, end_hour, y)
print('power_value', power_value, 'b_normal:', b_normal)

least_square(x, y)
w = get_model(x, y, regulation)