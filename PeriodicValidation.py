import time
import numpy as np

import keras
import tensorflow as tf
import keras.backend as K

from keras import optimizers
from keras.models import load_model
from keras.callbacks import Callback

from functions import calculate_top_k_new_only


"""
PeriodicValidation - Keras callback - checks val_loss periodically instead of using Model.fit() every epoch
"""
class PeriodicValidation(Callback):
    def __init__(self, val_data, batch_size, filepath):
        super(PeriodicValidation, self).__init__()
        self.val_data = val_data
        self.batch_size = batch_size
        self.filepath = filepath
        self.min_val_loss = np.Inf
    
    def on_epoch_end(self, epoch, logs={}):
        if epoch % 5 == 4 or epoch % 5 == 2:
            
            if self.filepath:
                self.model.save(self.filepath+".ep_"+str(epoch)+".h5", overwrite=True)
            
            if self.val_data is None:
                return
            
            h = self.model.evaluate(self.val_data[0], self.val_data[1], batch_size=self.batch_size, verbose=0)
            print("validating on " + str(self.val_data[1].shape[0]) + " samples on epoch " + str(epoch) + ": ", h)
            
            y_top_k_new_only = calculate_top_k_new_only(self.model, 
                self.val_data[0][0], self.val_data[0][1], self.val_data[1], self.batch_size, 
                (not self.val_data[0][1].shape[2] == self.val_data[1].shape[1]))
            print("testing MAP@K for NEW products: ", y_top_k_new_only)
            
            if h[0] < self.min_val_loss:
                if self.filepath:
                    self.model.save(self.filepath, overwrite=True)
                    print("val_loss improved from "+str(self.min_val_loss)+" to "+str(h[0])+", saving model to "+self.filepath)
                else:
                    print("val_loss improved from "+str(self.min_val_loss)+" to "+str(h[0]))
                self.min_val_loss = h[0]
    
    def on_train_end(self, logs=None): # also log training metrics with higher decimal precision
        print("epoch", [m for m in self.model.history.params['metrics']])
        for epoch in self.model.history.epoch:
            print(epoch, [self.model.history.history[m][epoch] for m in self.model.history.params['metrics']])

#
