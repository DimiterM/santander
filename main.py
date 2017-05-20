import time
import pandas as pd
import numpy as np

import dataset
from functions import *
from MergedModel import MergedModel

from keras import optimizers


print(time.strftime("%H:%M:%S", time.localtime()))
attr_cols = ["t", "sex", "age", "seniority", "is_primary", "is_domestic", "income"]
remove_non_buyers = False
A_buckets, X_buckets, y_buckets = dataset.load_trainset(cols=attr_cols, remove_non_buyers=remove_non_buyers)

for a, x, y in zip(A_buckets, X_buckets, y_buckets):
    print(a.shape, x.shape, y.shape)


print(time.strftime("%H:%M:%S", time.localtime()))
a_output_length = 24
x_output_length = 48#24
output_length = 24
input_dim = 26
attr_dim = 7
data_dim = 24#128
merged_data_dim = 12

model = MergedModel(
    a_output_length, x_output_length, output_length, 
    input_dim, attr_dim, 
    data_dim, merged_data_dim)
    
model.compile(
    loss=bin_crossentropy_true_only, 
    optimizer=optimizers.RMSprop(lr=0.001), 
    metrics=[in_top_k_loss, 'binary_crossentropy', 'mean_squared_error'])


print(time.strftime("%H:%M:%S", time.localtime()))
num_epochs = 100
batch_size = 256
model.train(A_buckets, X_buckets, y_buckets, num_epochs, batch_size)


print(time.strftime("%H:%M:%S", time.localtime()))
A_buckets, X_buckets, y_buckets = None, None, None
A_test_buckets, X_test_buckets, y_test_buckets, ids_test_buckets = dataset.load_testset(cols=attr_cols)

for a, x, y in zip(A_test_buckets, X_test_buckets, y_test_buckets):
    print(a.shape, x.shape, y.shape)

print(ids_test_buckets.shape)


print(time.strftime("%H:%M:%S", time.localtime()))
y_test_pred = model.predict(A_test_buckets, X_test_buckets, y_test_buckets, batch_size)
np.savetxt("./res.csv", np.concatenate((ids_test_buckets, y_test_pred), axis=1), delimiter=",")
