import time
import pandas as pd
import numpy as np

import dataset_baselines as dataset
from functions import *

from keras import optimizers

import keras.backend as K

from keras import optimizers
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, Callback

from keras.models import Model
from keras.layers.core import Dense
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Input

# from PeriodicValidation import PeriodicValidation
from functions import calculate_top_k_new_only_concatdataset



### set parameters
print(time.strftime("%H:%M:%S", time.localtime()))

train_month = 16
test_month = 17
attr_cols = ['t', 't_month', 'sex', 'age', 'seniority_new', 'seniority', 'is_primary', 'is_domestic', 'is_foreigner', 'is_dead', 'is_active', 'income', 'employee_bit_notN', 'country_num', 'customer_type_bit_not1', 'customer_rel_I', 'customer_rel_A', 'channel_0_0', 'channel_0_1', 'channel_0_2', 'channel_1_0', 'channel_1_1', 'channel_1_2', 'channel_1_3', 'channel_1_4', 'channel_1_5', 'channel_1_6', 'channel_1_7', 'channel_1_8', 'province_AFR', 'province_AND', 'province_ARA', 'province_AST', 'province_BAL', 'province_BAS', 'province_CAN', 'province_CAS', 'province_CAT', 'province_CNB', 'province_EXT', 'province_GAL', 'province_MAD', 'province_MAN', 'province_MUR', 'province_NAV', 'province_RIO', 'province_VAL', 'province_pop', 'segment_1', 'segment_2', 'segment_3']
remove_non_buyers = False
scale_time_dim = True
include_time_dim_in_X = True

save_models = False



### load train set
X_train, y_train, _ = dataset.load_concat_trainset(max_month=train_month, attr_cols=attr_cols, 
    remove_non_buyers=remove_non_buyers, scale_time_dim=scale_time_dim, include_time_dim_in_X=include_time_dim_in_X)

print(X_train.shape, y_train.shape)


### load test set
X_test, y_test, ids_test = dataset.load_concat_testset(train_month=train_month, test_month=test_month, attr_cols=attr_cols, 
    scale_time_dim=scale_time_dim, include_time_dim_in_X=include_time_dim_in_X)
print(X_test.shape, y_test.shape)
print(ids_test.shape)



output_length = 24
input_length = X_train.shape[1]

num_epochs = 40
batch_size = 256
# learning_rate = 0.0005#0.002


### create model
print(time.strftime("%H:%M:%S", time.localtime()))

model_input = Input(shape=(input_length,))
model = Dense(256)(model_input)
model = LeakyReLU()(model)
model = Dense(128)(model)
model = LeakyReLU()(model)
model = Dense(64)(model)
model = LeakyReLU()(model)
model = Dense(output_length, activation='sigmoid')(model)
model = Model(inputs=model_input, outputs=model)


model.compile(
    loss='binary_crossentropy', 
    optimizer='adam', 
    #metrics=[bin_crossentropy_true_only, in_top_k_loss, 'binary_crossentropy', 'mean_squared_error'])
    metrics=['binary_crossentropy', 'categorical_crossentropy', in_top_k_loss, 'mean_squared_error'])

print(time.strftime("%H:%M:%S", time.localtime()))
validation_data = (X_test, y_test) if test_month <= dataset.MAX_SEQUENCE_LENGTH else None
checkpoint_callback = ModelCheckpoint("./models/model_base_max_"+time.strftime("%m-%d_%H-%M", time.localtime())+".h5", 
    monitor="loss", save_best_only=True, verbose=1)
# periodic_val_callback = PeriodicValidation(validation_data, batch_size, 
#     ("./models/model_val_base_concat_"+time.strftime("%m-%d_%H-%M", time.localtime())+".h5") if save_models else None)
callbacks = ([checkpoint_callback] if save_models else []) #+ ([periodic_val_callback] if validation_data else [])
h = model.fit(X_train, y_train, batch_size, num_epochs, validation_data=validation_data, callbacks=callbacks, verbose=2)
print(h.params)

print(time.strftime("%H:%M:%S", time.localtime()))
###model.model.save("./model_base_concat_T"+str(train_month)+"_"+time.strftime("%m-%d_%H-%M", time.localtime())+".h5")


### score or predict
print(time.strftime("%H:%M:%S", time.localtime()))
if test_month <= dataset.MAX_SEQUENCE_LENGTH:
    # test data is from trainset
    print("testing score")
    print(time.strftime("%H:%M:%S", time.localtime()))
    model.evaluate(X_test, y_test, batch_size)
    print(time.strftime("%H:%M:%S", time.localtime()))
    
    y_top_k_new_only = calculate_top_k_new_only_concatdataset(model, X_test, y_test, batch_size, len(attr_cols))
    print("testing MAP@K for NEW products: ", y_top_k_new_only)
    
else:
    # test data is the testset
    print("predicting")
    y_test_pred = model.predict(A_test, X_test, batch_size)
    np.savetxt("./res_base_concat_T"+str(train_month)+"_"+time.strftime("%m-%d_%H-%M", time.localtime())+".csv", 
        np.concatenate((ids_test, y_test_pred), axis=1), delimiter=",")

#
