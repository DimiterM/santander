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
    attr_cols = ["t", "sex", "age", "seniority", "is_primary", "is_domestic"]##, "income"]
    remove_non_buyers = False
    
    A_buckets, X_buckets, y_buckets = dataset.load_trainset(max_month=last_month, cols=attr_cols, remove_non_buyers=remove_non_buyers)
    A_buckets, X_buckets, y_buckets = A_buckets[-1], X_buckets[-1], y_buckets[-1]
    
    A_test_buckets, X_test_buckets, y_test_buckets, _ = dataset.load_testset(month=last_month+1, cols=attr_cols)
    A_test_buckets, X_test_buckets, y_test_buckets = A_test_buckets[-1], X_test_buckets[-1], y_test_buckets[-1]
    
    return A_buckets, X_buckets, y_buckets, A_test_buckets, X_test_buckets, y_test_buckets

def merged_model(A_buckets, X_buckets, y_buckets, A_test_buckets, X_test_buckets, y_test_buckets):
    output_length = 24
    input_dim = 26
    attr_dim = 7

    a_input = Input(shape=(attr_dim,))
    a_model = Dense({{choice([24, 16, 32, 48])}}, activation={{choice(['relu', 'sigmoid', 'tanh'])}})(a_input)
    
    x_input = Input(shape=(None, input_dim))
    x_model = LSTM({{choice([24, 16, 32, 48])}},
        activation={{choice(['relu', 'sigmoid', 'tanh'])}}, recurrent_activation={{choice(['relu', 'sigmoid', 'tanh', 'hard_sigmoid'])}}, return_sequences=False)(x_input)
    
    if conditional({{choice(['one', 'two'])}}) == 'two':
        x_model = BatchNormalization()(x_model)
        x_model = Activation({{choice(['relu', 'sigmoid', 'tanh'])}})(x_model)
    
    x_model = Dropout({{uniform(0, 1)}})(x_model)
    x_model = Dense({{choice([24, 16, 32, 48, 64])}}, activation={{choice(['relu', 'sigmoid', 'tanh'])}})(x_model)
    
    model = keras.layers.concatenate([a_model, x_model])
    
    if conditional({{choice(['three', 'four'])}}) == 'four':
        model = Dense({{choice([24, 8, 12, 16, 32, 48])}}, activation={{choice(['relu', 'sigmoid', 'tanh'])}})(model)
    
    model = Dense(output_length, activation={{choice(['relu', 'sigmoid', 'tanh'])}})(model)
    
    model = Model(inputs=[a_input, x_input], outputs=model)
    
    
    model.compile(loss='binary_crossentropy', metrics=[bin_crossentropy_true_only, in_top_k_loss, 'binary_crossentropy', 'mean_squared_error'],
                  optimizer={{choice([optimizers.RMSprop(lr=0.001), optimizers.RMSprop(lr=0.005), optimizers.RMSprop(lr=0.01), 'adam', 'sgd'])}})

    model.fit([A_buckets, X_buckets], y_buckets,
              batch_size={{choice([128, 256, 512])}},
              epochs=15,
              verbose=2,
              validation_data=([A_test_buckets, X_test_buckets], y_test_buckets))
    score = model.evaluate([A_test_buckets, X_test_buckets], y_test_buckets, verbose=0)
    print('Test accuracy:', score[0])
    return {'loss': score[0], 'status': STATUS_OK, 'model': model}



if __name__ == '__main__':
    
    A_buckets, X_buckets, y_buckets, A_test_buckets, X_test_buckets, y_test_buckets = data()
    
    best_run, best_model = optim.minimize(model=merged_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=1,
                                          trials=Trials())
    
    print("Evalutation of best performing model:")
    print(best_model.evaluate([A_test_buckets, X_test_buckets], y_test_buckets))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)