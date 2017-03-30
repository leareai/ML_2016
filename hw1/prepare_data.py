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


def get_output(trunk, hrs):
    temp_array = np.array([data.ix[0, hrs:].values for data in trunk])
    temp_shape = temp_array.shape
    return temp_array.reshape(temp_shape[0]*temp_shape[1], )


def get_data(d_trunk, hrs):
    i = 0
    data_dict = {}
    for data in d_trunk:
        for j in range(data.columns.size - hrs):
            # vector = data.ix[:, j:j + hrs].values.T.flatten()
            vector = data.ix[:, j:j + hrs].values.flatten()
            data_dict[i] = vector
            i += 1

    input_data = DataFrame(data_dict).T
    normal_data = (input_data - input_data.mean()) / input_data.std()

    normal_data_base = normal_data
    normal_data_power = normal_data
    for j in range(1, power_value):
        normal_data_power = normal_data_power * normal_data_base
        normal_data = pd.concat([normal_data, normal_data_power], axis=1, ignore_index=True)

    normal_data['b'] = 1
    input_data['b'] = 1
    return normal_data, input_data

test_item = {
 u'AMB_TEMP',
 u'CH4',
 u'CO',
 u'NMHC',
 u'NO',
 u'NO2',
 u'NOx',
 u'O3',
 u'PM10',
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


if 'power_value' not in globals():
    power_value = 1

hours = 9

raw, raw_y = load_data(test_item)
dataTrunk = slice_into_trunks(raw, len(test_item))
y_trunk = slice_into_trunks(raw_y, 1)
y = get_output(y_trunk, hours)
x, origX = get_data(dataTrunk, hours)