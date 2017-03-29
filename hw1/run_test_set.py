import numpy as np
import pandas as pd
from pandas import Series, DataFrame


def load_test_data():
    test_raw_data = pd.read_csv('./data/test_X.csv', na_values='NR', index_col=0, header=None)
    test_raw_data.fillna(0, inplace=True)
    test_raw_data = test_raw_data.ix[test_raw_data[1].isin(test_item)]
    index = test_raw_data.index.unique()
    index.name = 'id'
    i = 0
    data_dict = {}
    for x in index:
        # vector = test_raw_data.ix[x, 2:].values.T.flatten()
        vector = test_raw_data.ix[x, 2:].values.flatten()
        data_dict[i] = vector
        i += 1

    input_data = DataFrame(data_dict).T
    normal_data = (input_data - input_data.mean()) / input_data.std()
    normal_data_base = normal_data
    normal_data_power = normal_data
    for j in range(1, power):
        normal_data_power = normal_data_power * normal_data_base
        normal_data = pd.concat([normal_data, normal_data_power], axis=1, ignore_index=True)
    normal_data['b'] = 1
    return index, normal_data, input_data

if 'test_item' not in globals():
    test_item = {u'AMB_TEMP', u'CH4', u'CO', u'NMHC', u'NO', u'NO2', u'NOx', u'O3',
                 u'PM10', u'PM2.5', u'RAINFALL', u'RH', u'SO2', u'THC', u'WD_HR',
                 u'WIND_DIREC', u'WIND_SPEED', u'WS_HR'}

if 'power' not in globals():
    power = 1

ids, testX, origTestX = load_test_data()
testXArray = testX.as_matrix();
testY = testXArray.dot(w)
result = DataFrame(testY, index=ids, columns=['value'])
result.to_csv('./result.csv')