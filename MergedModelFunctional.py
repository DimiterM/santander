import time
import numpy as np

import keras
import tensorflow as tf
import keras.backend as K

from keras import optimizers
from keras.models import load_model

from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Input, Merge

from sklearn.utils import shuffle

from ModelHistoryCheckpointer import ModelHistoryCheckpointer



class MergedModelFunctional:
    
    """
    output_length - .
    input_dim - .
    attr_dim - .

    a_hidden_length - .
    a_output_length - .

    recurrent_dim - .
    x_dropout_rate - .
    x_output_length - .

    merged_data_dim - .
    """
    def __init__(self, 
        output_length, input_dim, attr_dim, 
        a_hidden_length, a_output_length, 
        recurrent_dim, x_dropout_rate, x_output_length, 
        merged_data_dim):
        
        a_input = Input(shape=(attr_dim,))
        a_model = Dense(a_hidden_length, activation='softmax')(a_input)
        a_model = Dense(a_output_length, activation='softmax')(a_model)
        
        x_input = Input(shape=(None, input_dim))
        x_model = LSTM(recurrent_dim, activation='tanh', recurrent_activation='hard_sigmoid', return_sequences=False)(x_input)
        x_model = Dropout(x_dropout_rate)(x_model)
        x_model = Dense(x_output_length, activation='softmax')(x_model)
        
        self.model = keras.layers.concatenate([a_model, x_model])
        # self.model = Dense(merged_data_dim, activation='softmax')(self.model)
        self.model = Dense(output_length, activation='softmax')(self.model)
        
        self.model = Model(inputs=[a_input, x_input], outputs=self.model)
        self._a_model = a_model
        self._x_model = x_model
    
    
    def load_model_from_file(self, filename, custom_objects):
        self.model = load_model(filename, custom_objects=custom_objects)
        self._a_model = None
        self._x_model = None
    
    
    def compile(self, loss, optimizer, metrics):
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    
    
    def train_on_batch(self, A_buckets, X_buckets, y_buckets, num_epochs, batch_size, save_models=True):
        print("model.train_on_batch")
        self.checkpointer = ModelHistoryCheckpointer(self.model) if save_models else None
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
            
            if save_models:
                self.checkpointer.save_on_epoch(self.model, epoch, stats_all, batches_count)
        
        if save_models:
            self.checkpointer.save_last(self.model, epoch, stats_all, batches_count)
    
    
    def predict_on_batch(self, A_test_buckets, X_test_buckets, batch_size):
        print("model.predict_on_batch")
        y_test_pred = np.array([]).reshape(0, 24)
        for A_test, X_test in zip(A_test_buckets, X_test_buckets):
            if A_test.size > 0 and X_test.size > 0:
                y_pred = self.model.predict([A_test, X_test], batch_size=batch_size)
                y_test_pred = np.concatenate((y_test_pred, y_pred), axis=0)
        return y_test_pred
    
    
    def test_on_batch(self, A_test_buckets, X_test_buckets, y_test_buckets):
        print("model.test_on_batch")
        stats_all = np.zeros(len(self.model.metrics)+1)
        examples_count = 0
        for A_test, X_test, y_test in zip(A_test_buckets, X_test_buckets, y_test_buckets):
            for A, X, y in zip(A_test, X_test, y_test):
                stats = self.model.test_on_batch(x=[np.array([A]), np.array([X])], y=np.array([y])) # batch of 1
                stats_all = stats_all + stats
                examples_count += 1
            print("--> ", stats_all, examples_count)
        print(stats_all, examples_count)
    
    
    def train(self, A_train, X_train, y_train, num_epochs, batch_size, save_models=True):
        if type(X_train).__name__ == "list":
            self.train_on_batch(A_train, X_train, y_train, num_epochs, batch_size, save_models)
        else: # X_train is NumPy array
            self.model.fit([A_train, X_train], y_train, batch_size, num_epochs)
    
    
    def predict(self, A_test, X_test, batch_size):
        if type(X_test).__name__ == "list":
            return self.predict_on_batch(A_test, X_test, batch_size)
        # X_test is NumPy array
        return self.model.predict([A_test, X_test], batch_size)
    
    
    def test(self, A_test, X_test, y_test):
        if type(X_test).__name__ == "list":
            self.test_on_batch(A_test, X_test, y_test)
        else: # X_test is NumPy array
            h = self.model.evaluate([A_test, X_test], y_test, batch_size=y_test.shape[0])
            print(h)



# 
