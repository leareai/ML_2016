import numpy as np
import pandas as pd
from pandas import Series, DataFrame


def fix_value(feature, index):
    # print('fix>>', feature)
    length = len(feature)
    if index == 0:
        feature[0] = 2 * feature[1] - feature[2]
    elif index == length - 1:
        feature[8] = 2 * feature[7] - feature[6]
    else:
        left = next(i for i in range(index, -1, -1) if i <= 0 or feature[i] > 0)
        right = next(i for i in range(index, length) if i >= length or feature[i] > 0)
        delta = (feature[right] - feature[left])/(right - left);
        prev = feature[left];
        for idx in range(left+1, right):
            prev = feature[idx] = prev + delta;
    # print('fix<<', feature)


def load_test_data(start_hr):
    global origX, b_normal
    test_raw_data = pd.read_csv('./data/test_X.csv', na_values='NR', index_col=0, header=None)
    test_raw_data.fillna(0, inplace=True)
    test_raw_data = test_raw_data.ix[test_raw_data[1].isin(test_item)]
    index = test_raw_data.index.unique()
    index.name = 'id'
    i = 0
    data_dict = {}
    hour_len = 9 - start_hour;
    for xx in index:
        # vector = test_raw_data.ix[xx, 2+start_hr:].values
        vector = test_raw_data.ix[xx, 2:].values
        [fix_value(yy, count, ) for yy in vector for count in range(hour_len) if yy[count] < 0]
        vector = vector[:, start_hr:]
        vector = vector.flatten()
        data_dict[i] = vector
        i += 1

    input_data = DataFrame(data_dict).T
    if (b_normal):
        normal_data = (input_data - origX.mean()) / origX.std()
    else:
        normal_data = input_data.copy()
    normal_data_base = normal_data
    normal_data_power = normal_data
    for j in range(1, power_value):
        normal_data_power = normal_data_power * normal_data_base
        normal_data = pd.concat([normal_data, normal_data_power], axis=1, ignore_index=True)
    normal_data['b'] = 1
    return index, normal_data, input_data

if 'b_normal' not in globals():
    b_normal = False;

if 'test_item' not in globals():
    test_item = {u'AMB_TEMP', u'CH4', u'CO', u'NMHC', u'NO', u'NO2', u'NOx', u'O3',
                 u'PM10', u'PM2.5', u'RAINFALL', u'RH', u'SO2', u'THC', u'WD_HR',
                 u'WIND_DIREC', u'WIND_SPEED', u'WS_HR'}

if 'power_value' not in globals():
    power_value = 1

if 'start_hour' not in globals():
    start_hour = 0

print('power_value', power_value, 'b_normal:',b_normal, 'start_hour:', start_hour)
print(test_item)

ids, testX, origTestX = load_test_data(start_hour)
testXArray = testX.as_matrix();
testY = testXArray.dot(w)
result = DataFrame(testY, index=ids, columns=['value'])
result.to_csv('./result.csv')
print ('save result!!')
