import time
import numpy as np

import keras
import tensorflow as tf
import keras.backend as K

from keras import optimizers
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, Callback

from keras.models import Model
from keras.layers.recurrent import LSTM, GRU
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Input, Merge, Masking
from keras.layers.wrappers import Bidirectional

from sklearn.utils import shuffle

from ModelHistoryCheckpointer import ModelHistoryCheckpointer
from PeriodicValidation import PeriodicValidation
from functions import calculate_top_k_new_only



class RecurrentModel:
    
    """
    time_dim -> None or timesteps dimension size
    output_length -> number of classes
    input_dim -> X dataset length
    attr_dim -> A dataset length

    a_hidden_length -> hidden layer size for a_model branch
    a_output_length -> output layer size for a_model branch

    recurrent_dim -> recurrent layer size (for x_model branch)
    rnn_architecture -> "lstm" or "gru"
    go_direction -> 1 (forward), -1 (backward), 2 (bidirectional)
    
    dropout_rate -> dropout rate
    x_output_length -> output layer size for x_model branch
    
    merged_data_dim -> last hidden layer size
    """
    def __init__(self, time_dim, 
        output_length, input_dim, attr_dim, 
        a_hidden_length, a_output_length, 
        recurrent_dim, rnn_architecture, go_direction, dropout_rate, x_output_length, 
        merged_data_dim):
        
        a_input = Input(shape=(attr_dim,))
        a_model = Dense(a_hidden_length)(a_input)
        a_model = LeakyReLU()(a_model)
        a_model = Dense(a_output_length)(a_model)
        a_model = LeakyReLU()(a_model)
        
        
        x_input = Input(shape=(time_dim, input_dim))
        x_model = Masking(mask_value=-1.0)(x_input)
        
        RNN_Architecture = GRU if rnn_architecture == "gru" else LSTM
        if go_direction in [-1, 1]:
            x_model = RNN_Architecture(recurrent_dim, activation='tanh', recurrent_activation='hard_sigmoid', 
                return_sequences=False, go_backwards=(go_direction == -1))(x_model)
        else: # go_direction == 2
            x_model = Bidirectional(
                RNN_Architecture(recurrent_dim, activation='tanh', recurrent_activation='hard_sigmoid', return_sequences=False),
                merge_mode='concat')(x_model)
        
        x_model = Dropout(dropout_rate)(x_model)
        x_model = Dense(x_output_length)(x_model)
        x_model = LeakyReLU()(x_model)
        
        
        self.model = keras.layers.concatenate([a_model, x_model])
        self.model = Dropout(dropout_rate)(self.model)
        
        if merged_data_dim > 0:
            self.model = Dense(merged_data_dim)(self.model)
            self.model = LeakyReLU()(self.model)
        
        self.model = Dense(output_length, activation='sigmoid')(self.model)
        
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
        print(stats_all / examples_count)
    
    
    def train(self, A_train, X_train, y_train, num_epochs, batch_size, validation_data=None, save_models=True):
        if type(X_train).__name__ == "list":
            self.train_on_batch(A_train, X_train, y_train, num_epochs, batch_size, save_models)
        else: # X_train is NumPy array
            checkpoint_callback = ModelCheckpoint("./models/model_"+time.strftime("%m-%d_%H-%M", time.localtime())+".h5", 
                monitor="loss", save_best_only=True, verbose=1)
            # lr_callback = ReduceLROnPlateau(monitor="loss", 
            #     factor=0.5, patience=5, verbose=1, mode="auto", epsilon=0.0001, cooldown=0, min_lr=0.0001)
            periodic_val_callback = PeriodicValidation(validation_data, batch_size, 
                ("./models/model_val_"+time.strftime("%m-%d_%H-%M", time.localtime())+".h5") if save_models else None)
            # callbacks = [lr_callback] + ([checkpoint_callback] if save_models else []) + ([periodic_val_callback] if validation_data else [])
            callbacks = ([checkpoint_callback] if save_models else []) + ([periodic_val_callback] if validation_data else [])
            h = self.model.fit([A_train, X_train], y_train, batch_size, num_epochs, validation_data=None, callbacks=callbacks, verbose=2)
            print(h.params)
            # print("training history: ", h.params, h.history)
    
    
    def predict(self, A_test, X_test, batch_size):
        if type(X_test).__name__ == "list":
            return self.predict_on_batch(A_test, X_test, batch_size)
        # X_test is NumPy array
        return self.model.predict([A_test, X_test], batch_size)
    
    
    def test(self, A_test, X_test, y_test, batch_size):
        if type(X_test).__name__ == "list":
            h = self.test_on_batch(A_test, X_test, y_test)
            print("test_on_batch history: ", h)
        else: # X_test is NumPy array
            h = self.model.evaluate([A_test, X_test], y_test, batch_size=batch_size, verbose=1)
            print("testing history: ", h)



# 
