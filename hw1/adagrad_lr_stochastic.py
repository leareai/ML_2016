# coding:utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

import sys
from pandas import Series, DataFrame
from random import randint

import signal

def set_signal(log_func):
    def signal_handler(sig, frame):
        print('\n')
        log_func()
        print('time:', (time.time() - start_time))
        if sig == signal.SIGQUIT:
            print('\nYou pressed Ctrl+\\')
            global loggable
            loggable = not loggable
        elif sig == signal.SIGINT:
            print('\nYou pressed Ctrl+C:')
            sys.exit(0)

    signal.signal(signal.SIGQUIT, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)


def stochastic_gradient_descent_with_adagrad(input_x, output_y, base_rate=0.1, batch_size_value=256):
    global w, loss, e
    min_gradient_value = 1e-6
    min_descent = e-7
    gradient_value = float('Inf')
    descent = float('Inf')
    rate = base_rate
    count = 0
    data_size = input_x.shape[0]

    fudge_factor = 1e-6
    feature_number = input_x.shape[1]
    adagrad = np.zeros(feature_number);

    def log():
        global e, loss
        e = output_y - input_x.dot(w)
        e_square = e.dot(e)
        target_loss = np.sqrt((e_square + regulation * w.dot(w))/data_size)
        adagrad_value = np.sqrt(adagrad.dot(adagrad))
        print('grad:', gradient_value, 'adagrad', adagrad_value, 'decent:', descent, ', target loss:', target_loss, ', count:', count)


    set_signal(log)

    current_index = data_size
    io_pair = np.c_[input_x.reshape(data_size, -1), output_y.reshape(data_size, -1)]
    gradient_square_sum = np.zeros(feature_number)
    while gradient_value > min_gradient_value and descent > min_descent:
        if current_index >= data_size:
            np.random.shuffle(io_pair)
            current_index = 0
            input_x = io_pair[:, :feature_number]
            output_y = io_pair[:, -1]

        start = current_index
        current_index += batch_size_value
        x_picked = input_x[start:current_index]
        y_picked = output_y[start:current_index]
        e_picked = y_picked - x_picked.dot(w)
        gradient = -2 * e_picked.dot(x_picked) + 2 * regulation * w

        gradient_square_sum += gradient * gradient
        adagrad = np.sqrt(gradient_square_sum+fudge_factor)
        adagrad_gradient = rate * gradient / adagrad

        w -= adagrad_gradient
        descent = np.sqrt(adagrad_gradient.dot(adagrad_gradient))
        gradient_value = np.sqrt(gradient.dot(gradient))
        if loggable and count % log_rate == 0:
            log()
        count += 1

    log()
    return w, loss

w = np.ones(x.columns.size)
loss = e = 0

if 'loggable' not in globals():
    loggable = False

if 'batch_size' not in globals():
    batch_size = 64

if 'log_rate' not in globals():
    log_rate = 100000/batch_size

if 'adagrad_base_rate' not in globals():
    adagrad_base_rate = 0.1

if 'regulation' not in globals():
    regulation = 0;

print('Start iteration, you can pressed Ctrl+\\ to switch log, pressed Ctrl+C to force terminate')
print ('base rate:', adagrad_base_rate, 'regulation:', regulation, 'batch_size:', batch_size)

start_time = time.time()
w, loss = stochastic_gradient_descent_with_adagrad(x.as_matrix(), y, adagrad_base_rate, batch_size)
print ('time:', (time.time() - start_time))