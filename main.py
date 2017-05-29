import time
import pandas as pd
import numpy as np

import dataset
from functions import *
from MergedModel import MergedModel
from MergedModelFunctional import MergedModelFunctional

from keras import optimizers


print(time.strftime("%H:%M:%S", time.localtime()))
last_month = 16
# attr_cols = ["t", "sex", "age", "seniority", "is_primary", "is_domestic", "income"]
attr_cols = ['t', 't_month', 'sex', 'age', 'seniority_new', 'seniority', 'is_primary', 'is_domestic', 'is_foreigner', 'is_dead', 'is_active', 'income', 'employee_bit_notN', 'country_num', 'customer_type_bit_not1', 'customer_rel_I', 'customer_rel_A', 'channel_0_0', 'channel_0_1', 'channel_0_2', 'channel_1_0', 'channel_1_1', 'channel_1_2', 'channel_1_3', 'channel_1_4', 'channel_1_5', 'channel_1_6', 'channel_1_7', 'channel_1_8', 'segment_1', 'segment_2', 'segment_3']
remove_non_buyers = False
A_buckets, X_buckets, y_buckets = dataset.load_trainset(max_month=last_month, attr_cols=attr_cols, remove_non_buyers=remove_non_buyers)

for a, x, y in zip(A_buckets, X_buckets, y_buckets):
    print(a.shape, x.shape, y.shape)


print(time.strftime("%H:%M:%S", time.localtime()))
a_output_length = 24
x_dropout_rate = 0.2
x_output_length = 48#24
output_length = 24
input_dim = 26
attr_dim = len(attr_cols)
data_dim = 24#128
merged_data_dim = 12

model = MergedModelFunctional(
    a_output_length, x_dropout_rate, x_output_length, output_length, 
    input_dim, attr_dim, 
    data_dim, merged_data_dim)
    
model.compile(
    loss='binary_crossentropy', 
    optimizer=optimizers.RMSprop(lr=0.001), 
    metrics=[bin_crossentropy_true_only, in_top_k_loss, 'binary_crossentropy', 'mean_squared_error'])


print(time.strftime("%H:%M:%S", time.localtime()))
num_epochs = 30
batch_size = 256
model.train(A_buckets, X_buckets, y_buckets, num_epochs, batch_size)


print(model.checkpointer.file_index_log)
print(time.strftime("%H:%M:%S", time.localtime()))
A_buckets, X_buckets, y_buckets = None, None, None
A_test_buckets, X_test_buckets, y_test_buckets, ids_test_buckets = dataset.load_testset(month=last_month+1, attr_cols=attr_cols)

for a, x, y in zip(A_test_buckets, X_test_buckets, y_test_buckets):
    print(a.shape, x.shape, y.shape)

print(ids_test_buckets.shape)


print(time.strftime("%H:%M:%S", time.localtime()))
if last_month < dataset.MAX_SEQUENCE_LENGTH:
    # test data is from trainset
    print("testing score")
    model.test(A_test_buckets, X_test_buckets, y_test_buckets)
else:
    # test data is the testset
    print("predicting")
    y_test_pred = model.predict(A_test_buckets, X_test_buckets, batch_size)
    np.savetxt("./res10.csv", np.concatenate((ids_test_buckets, y_test_pred), axis=1), delimiter=",")

print(time.strftime("%H:%M:%S", time.localtime()))
model.model.save("./mmf_"+time.strftime("%m-%d_%H-%M_", time.localtime())+".h5")
