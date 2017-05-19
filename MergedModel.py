import time
import numpy as np

import keras
import tensorflow as tf
import keras.backend as K

from keras import optimizers

from keras.models import Sequential

from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Merge

from sklearn.utils import shuffle


class MergedModel:

    """
    a_output_length - .
    x_output_length - .
    output_length - .
    input_dim - .
    attr_dim - .
    data_dim - .
    merged_data_dim - .
    """
    def __init__(self, 
        a_output_length, x_output_length, 
        output_length, input_dim, 
        attr_dim, data_dim, merged_data_dim
        ):
        a_input = Sequential()
        a_input.add(Dense(a_output_length, activation='softmax', input_dim=attr_dim))

        x_input = Sequential()
        x_input.add(LSTM(data_dim, input_dim=input_dim,
            activation='sigmoid', inner_activation='hard_sigmoid', return_sequences=False))
        # x_input.add(BatchNormalization()) #
        # x_input.add(Activation('sigmoid')) #
        x_input.add(Dropout(0.2))
        x_input.add(Dense(x_output_length, activation='softmax'))

        self.model = Sequential()
        self.model.add(Merge([a_input, x_input], mode='concat'))
        # self.model.add(Dense(merged_data_dim, activation='softmax'))
        self.model.add(Dense(output_length, activation='softmax'))


    def compile(self, loss, optimizer, metrics):
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)


    def train(self, A_buckets, X_buckets, y_buckets, num_epochs, batch_size):
        for epoch in range(num_epochs):
            stats_all = np.zeros(len(self.model.metrics)+1)
            batches_count = 0
            print("epoch: " + str(epoch) + " // " + time.strftime("%H:%M:%S", time.localtime()))
            for A_train, X_train, y_train in zip(A_buckets, X_buckets, y_buckets):
                A_train, X_train, y_train = shuffle(A_train, X_train, y_train)
                batch_indices_or_sections = [i * batch_size for i in range(1, len(X_train) // batch_size)]
                A_train_batches = np.array_split(A_train, batch_indices_or_sections)
                X_train_batches = np.array_split(X_train, batch_indices_or_sections)
                y_train_batches = np.array_split(y_train, batch_indices_or_sections)
                for A_batch, X_batch, y_batch in zip(A_train_batches, X_train_batches, y_train_batches):
                    stats = self.model.train_on_batch(x=[A_batch, X_batch], y=y_batch)
                    stats_all = stats_all + stats
                    batches_count += 1
                if(epoch % 10 == 9):
                    print(stats)
            print(stats_all, batches_count)


    def predict(self, A_test_buckets, X_test_buckets, y_test_buckets, batch_size):
        y_test_pred = np.array([]).reshape(0, 24)
        for A_test, X_test, y_test in zip(A_test_buckets, X_test_buckets, y_test_buckets):
            if A_test.size > 0 and X_test.size > 0 and y_test.size > 0:
                y_pred = self.model.predict([A_test, X_test], batch_size=batch_size)
                y_test_pred = np.concatenate((y_test_pred, y_pred), axis=0)
        return y_test_pred

