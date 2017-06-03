from __future__ import print_function

from hyperopt import Trials, STATUS_OK, tpe
from keras.datasets import mnist
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from keras.utils import np_utils

from hyperas import optim
from hyperas.distributions import choice, uniform, conditional

from functions import *
import dataset

import time
import numpy as np

import keras
import tensorflow as tf
import keras.backend as K

from keras import optimizers

from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Input, Merge


def data():
    last_month = 16
    # attr_cols = ["t", "sex", "age", "seniority", "is_primary", "is_domestic", "income"]
    attr_cols = ['t', 't_month', 'sex', 'age', 'seniority_new', 'seniority', 'is_primary', 'is_domestic', 'is_foreigner', 'is_dead', 'is_active', 'income', 
        'employee_bit_notN', 'country_num', 'customer_type_bit_not1', 'customer_rel_I', 'customer_rel_A', 'channel_0_0', 'channel_0_1', 'channel_0_2', 
        'channel_1_0', 'channel_1_1', 'channel_1_2', 'channel_1_3', 'channel_1_4', 'channel_1_5', 'channel_1_6', 'channel_1_7', 'channel_1_8', 
        'segment_1', 'segment_2', 'segment_3']
    remove_non_buyers = False
    scale_time_dim = True
    include_time_dim_in_X = True
    
    A_buckets, X_buckets, y_buckets = dataset.load_trainset(max_month=last_month, attr_cols=attr_cols, 
        remove_non_buyers=remove_non_buyers, scale_time_dim=scale_time_dim, include_time_dim_in_X=include_time_dim_in_X)
    A_buckets, X_buckets, y_buckets = A_buckets[-1], X_buckets[-1], y_buckets[-1]
    
    A_test_buckets, X_test_buckets, y_test_buckets, _ = dataset.load_testset(last_month=last_month, next_month=last_month+1, attr_cols=attr_cols, 
        scale_time_dim=scale_time_dim, include_time_dim_in_X=include_time_dim_in_X)
    A_test_buckets, X_test_buckets, y_test_buckets = A_test_buckets[-1], X_test_buckets[-1], y_test_buckets[-1]
    
    return A_buckets, X_buckets, y_buckets, A_test_buckets, X_test_buckets, y_test_buckets


def merged_model(A_buckets, X_buckets, y_buckets, A_test_buckets, X_test_buckets, y_test_buckets):
    output_length = 24 # NUM_CLASSES
    input_dim = 26 # NUM_CLASSES or NUM_CLASSES + 2
    attr_dim = 32 # len(attr_cols)

    a_input = Input(shape=(attr_dim,))
    a_model = Dense(64, activation='relu')(a_input)
    a_model = Dense(16, activation='softmax')(a_input)
    
    x_input = Input(shape=(None, input_dim))
    x_model = LSTM(48, activation='tanh', recurrent_activation='hard_sigmoid', return_sequences=False)(x_input)
    x_model = Dropout(0.1)(x_model)
    x_model = Dense(32, activation={{choice(['relu', 'sigmoid', 'tanh', 'softmax'])}})(x_model)
    model = keras.layers.concatenate([a_model, x_model])
    
    # if conditional({{choice(['three', 'four'])}}) == 'four':
    #     model = Dense({{choice([8, 12, 16, 24, 32, 48])}}, activation='sigmoid')(model)
    
    model = Dense(output_length, activation='sigmoid')(model)
    model = Model(inputs=[a_input, x_input], outputs=model)
     
    model.compile(loss='binary_crossentropy', metrics=[bin_crossentropy_true_only, in_top_k_loss, 'binary_crossentropy', 'mean_squared_error'],
        optimizer=choice([optimizers.RMSprop(lr=0.001))

    model.fit([A_buckets, X_buckets], y_buckets,
        batch_size={{choice([256, 512])}},
        epochs=15,
        verbose=2,
        validation_data=([A_test_buckets, X_test_buckets], y_test_buckets))
    score = model.evaluate([A_test_buckets, X_test_buckets], y_test_buckets, verbose=0)
    print('Test accuracy:', score[0])
    return {'loss': score[0], 'status': STATUS_OK, 'model': model}



if __name__ == '__main__':
    
    A_buckets, X_buckets, y_buckets, A_test_buckets, X_test_buckets, y_test_buckets = data()
    
    trials = Trials()
    best_run, best_model = optim.minimize(model=merged_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=10,
                                          trials=trials)
    
    print("Evalutation of best performing model:")
    print(best_model.evaluate([A_test_buckets, X_test_buckets], y_test_buckets))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
    print("Trials object:")
    print(trials)
    print(trials.trials)
    print(trials.results)
    print(trials.losses())
    print(trials.statuses())

