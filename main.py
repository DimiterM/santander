import time
import pandas as pd
import numpy as np

import dataset
from functions import *
from RecurrentModel import RecurrentModel

from keras import optimizers


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--trainset', default="./catdf.csv")
parser.add_argument('--testset', default="./testcatdf.csv")
parser.add_argument('-m', '--train_month', type=int, default=17)
parser.add_argument('-t', '--test_month', type=int, default=18)
parser.add_argument('--use_buckets', action="store_true", default=False)
parser.add_argument('-f', '--model_filename')
parser.add_argument('-a', '--rnn_architecture', default="lstm")
parser.add_argument('-g', '--go_direction', type=int, default=1)
parser.add_argument('-n', '--num_epochs', type=int, default=20)
parser.add_argument('-b', '--batch_size', type=int, default=256)
parser.add_argument('-l', '--learning_rate', type=float, default=0.001)
args = parser.parse_args()


trainset_filename = args.trainset or "./catdf.csv"
testset_filename = args.testset or "./testcatdf.csv"
print(trainset_filename, testset_filename)


### set parameters
print(time.strftime("%H:%M:%S", time.localtime()))
load_model_filename = args.model_filename or None

train_month = args.train_month or 17
test_month = args.test_month or 18
attr_cols = ['t', 't_month', 'sex', 'age', 'seniority_new', 'seniority', 'is_primary', 'is_domestic', 'is_foreigner', 'is_dead', 'is_active', 'income', 'employee_bit_notN', 'country_num', 'customer_type_bit_not1', 'customer_rel_I', 'customer_rel_A', 'channel_0_0', 'channel_0_1', 'channel_0_2', 'channel_1_0', 'channel_1_1', 'channel_1_2', 'channel_1_3', 'channel_1_4', 'channel_1_5', 'channel_1_6', 'channel_1_7', 'channel_1_8', 'province_AFR', 'province_AND', 'province_ARA', 'province_AST', 'province_BAL', 'province_BAS', 'province_CAN', 'province_CAS', 'province_CAT', 'province_CNB', 'province_EXT', 'province_GAL', 'province_MAD', 'province_MAN', 'province_MUR', 'province_NAV', 'province_RIO', 'province_VAL', 'province_pop', 'segment_1', 'segment_2', 'segment_3']
remove_non_buyers = False
scale_time_dim = True
include_time_dim_in_X = True
use_fixed_seq_len = not args.use_buckets # True

output_length = 24
input_dim = dataset.NUM_CLASSES + 2 if include_time_dim_in_X else dataset.NUM_CLASSES
attr_dim = len(attr_cols)
rnn_architecture = args.rnn_architecture or "lstm"
go_direction = args.go_direction or 1
a_hidden_length = 60
a_output_length = 24
recurrent_dim = 48
x_output_length = 48
dropout_rate = 0.1
merged_data_dim = 50

num_epochs = args.num_epochs or 20
batch_size = args.batch_size or 256
learning_rate = args.learning_rate or 0.001



### load train set
A_train, X_train, y_train = None, None, None
if load_model_filename is None or load_model_filename == "":
    if not use_fixed_seq_len:
        A_train, X_train, y_train = dataset.load_trainset(trainset_filename=trainset_filename, 
            max_month=train_month, attr_cols=attr_cols, 
            remove_non_buyers=remove_non_buyers, scale_time_dim=scale_time_dim, include_time_dim_in_X=include_time_dim_in_X)
        
        for a, x, y in zip(A_train, X_train, y_train):
            print(a.shape, x.shape, y.shape)
        
    else:
        A_train, X_train, y_train = dataset.load_padded_trainset(trainset_filename=trainset_filename, 
            max_month=train_month, attr_cols=attr_cols, 
            remove_non_buyers=remove_non_buyers, scale_time_dim=scale_time_dim, include_time_dim_in_X=include_time_dim_in_X, seq_len=17)
        
        print(A_train.shape, X_train.shape, y_train.shape)


### load test set
if not use_fixed_seq_len:
    A_test, X_test, y_test, ids_test = dataset.load_testset(trainset_filename=trainset_filename, testset_filename=testset_filename, 
        train_month=train_month, test_month=test_month, attr_cols=attr_cols, 
        scale_time_dim=scale_time_dim, include_time_dim_in_X=include_time_dim_in_X)
    
    for a, x, y in zip(A_test, X_test, y_test):
        print(a.shape, x.shape, y.shape)
    
    print(ids_test.shape)
    
else:
    A_test, X_test, y_test, ids_test = dataset.load_padded_testset(trainset_filename=trainset_filename, testset_filename=testset_filename, 
        train_month=train_month, test_month=test_month, attr_cols=attr_cols, 
        scale_time_dim=scale_time_dim, include_time_dim_in_X=include_time_dim_in_X)
    print(A_test.shape, X_test.shape, y_test.shape)
    print(ids_test.shape)


if load_model_filename is not None and load_model_filename != "":
    ### load model
    print("loading model from file: " + load_model_filename)
    model = RecurrentModel(None,
        output_length, input_dim, attr_dim, 
        a_hidden_length, a_output_length, 
        recurrent_dim, rnn_architecture, go_direction, dropout_rate, x_output_length, 
        merged_data_dim)
    model.load_model_from_file(load_model_filename, custom_objects={"in_top_k_loss": in_top_k_loss, "bin_crossentropy_true_only": bin_crossentropy_true_only})
    
else:
    ### create model
    print(time.strftime("%H:%M:%S", time.localtime()))
    
    model = RecurrentModel(None,
        output_length, input_dim, attr_dim, 
        a_hidden_length, a_output_length, 
        recurrent_dim, rnn_architecture, go_direction, dropout_rate, x_output_length, 
        merged_data_dim)
    
    model.compile(
        loss='binary_crossentropy', 
        optimizer=optimizers.RMSprop(lr=learning_rate), 
        #metrics=[bin_crossentropy_true_only, in_top_k_loss, 'binary_crossentropy', 'mean_squared_error'])
        metrics=['binary_crossentropy', 'categorical_crossentropy', in_top_k_loss, 'mean_squared_error'])
    
    print(time.strftime("%H:%M:%S", time.localtime()))
    validation_data = ([A_test, X_test], y_test) if test_month <= dataset.MAX_SEQUENCE_LENGTH else None
    model.train(A_train, X_train, y_train, num_epochs, batch_size, validation_data=validation_data, save_models=True)
    
    print(model.checkpointer.file_index_log if hasattr(model, "checkpointer") else "")
    print(time.strftime("%H:%M:%S", time.localtime()))
    model.model.save("./model_T"+str(train_month)+"_"+time.strftime("%m-%d_%H-%M", time.localtime())+".h5")


### score or predict
print(time.strftime("%H:%M:%S", time.localtime()))
if test_month <= dataset.MAX_SEQUENCE_LENGTH:
    # test data is from trainset
    print("testing score")
    print(time.strftime("%H:%M:%S", time.localtime()))
    model.test(A_test, X_test, y_test, batch_size)
    print(time.strftime("%H:%M:%S", time.localtime()))
    
    y_top_k_new_only = calculate_top_k_new_only(model, A_test, X_test, y_test, batch_size, include_time_dim_in_X)
    print("testing MAP@K for NEW products: ", y_top_k_new_only)
    
else:
    # test data is the testset
    print("predicting")
    y_test_pred = model.predict(A_test, X_test, batch_size)
    np.savetxt("./res_T"+str(train_month)+"_"+time.strftime("%m-%d_%H-%M", time.localtime())+".csv", 
        np.concatenate((ids_test, y_test_pred), axis=1), delimiter=",")

#
