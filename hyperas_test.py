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
        'province_AFR', 'province_AND', 'province_ARA', 'province_AST', 'province_BAL', 'province_BAS', 
        'province_CAN', 'province_CAS', 'province_CAT', 'province_CNB', 'province_EXT', 'province_GAL', 'province_MAD', 
        'province_MAN', 'province_MUR', 'province_NAV', 'province_RIO', 'province_VAL', 'province_pop', 
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
    attr_dim = 51 # len(attr_cols)

    a_hidden_length = 72
    a_output_length = 24  
    recurrent_dim = 48
    x_dropout_rate = 0.1
    x_output_length = 48#24
    merged_data_dim = 16

    a_input = Input(shape=(attr_dim,))
    a_model = Dense(a_hidden_length, activation='softmax')(a_input)
    a_model = Dense(a_output_length, activation='softmax')(a_model)
    
    x_input = Input(shape=(None, input_dim))
    x_model = LSTM(recurrent_dim, activation='tanh', recurrent_activation='hard_sigmoid', return_sequences=False)(x_input)
    x_model = Dropout(x_dropout_rate)(x_model)
    x_model = Dense(x_output_length, activation='softmax')(x_model)
    
    model = keras.layers.concatenate([a_model, x_model])
    # model = Dense(merged_data_dim, activation='softmax')(model)
    model = Dense(output_length, activation='softmax')(model)
    
    model = Model(inputs=[a_input, x_input], outputs=model)
     
    model.compile(loss='binary_crossentropy', metrics=[bin_crossentropy_true_only, in_top_k_loss, 'binary_crossentropy', 'mean_squared_error'],
        optimizer={{choice([optimizers.RMSprop(lr=0.001), optimizers.RMSprop(lr=0.0001), optimizers.RMSprop(lr=0.0005)])}})

    model.fit([A_buckets, X_buckets], y_buckets,
        batch_size={{choice([256, 512])}},
        epochs=30,
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
                                          max_evals=6,#10,
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

