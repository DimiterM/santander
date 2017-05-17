import time
import pandas as pd
import numpy as np
print(time.strftime("%H:%M:%S", time.localtime()))
df = pd.read_csv("./df.csv")

df_counts = df[["id", "t"]].groupby("id").count()
df_counts["id"] = df_counts.index
df_counts.columns = ["t_count", "id"]

df_maxs = df[["id", "t"]].groupby("id").max()
df_maxs["id"] = df_maxs.index
df_maxs.columns = ["t_max", "id"]

df = pd.merge(df_counts, df_maxs, how="outer", on=["id"]).merge(df, how="outer", on=["id"])

MAX_SEQUENCE_LENGTH = 17
df_tmax_groups = []
for i in range(2, MAX_SEQUENCE_LENGTH + 1):
    df_tmax_groups.append(df.loc[df["t_count"] == i].sort_values(["id", "t"]))

df_attr_groups = []
for i in range(2, MAX_SEQUENCE_LENGTH + 1):    # TODO: transform attrs...
    df_attr_groups.append(df.loc[(df["t_count"] == i) & (df["t_max"] == df["t"])].sort_values(["id"])[ \
        ["t", "sex", "age", "seniority", "is_primary", "is_domestic", "income"]] \
        .fillna(value=0))

ys = df.columns.tolist()[-24:]
df = None

for i in range(len(df_tmax_groups)):
    df_tmax_groups[i] = df_tmax_groups[i].loc[:, ['t', 't_month']+ys].as_matrix()
    df_tmax_groups[i] = df_tmax_groups[i].reshape(df_tmax_groups[i].shape[0]//(i+2), i+2, len(['t', 't_month']+ys))
    df_attr_groups[i] = df_attr_groups[i].as_matrix()

X_buckets = []
y_buckets = []
for g in df_tmax_groups:
    X_buckets.append(g[:, :-1, :])
    y_buckets.append(g[:, -1:, 2:].reshape(g.shape[0], g.shape[2] - 2))

df_tmax_groups = None
A_buckets = df_attr_groups
df_attr_groups = None

for a, x, y in zip(A_buckets, X_buckets, y_buckets):
    print(a.shape, x.shape, y.shape)




print(time.strftime("%H:%M:%S", time.localtime()))
import keras
import tensorflow as tf
import keras.backend as K
from keras import optimizers
# sess = tf.Session()
# sess = tf.InteractiveSession()
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Merge

from sklearn.utils import shuffle

def bin_crossent_true_only(y_true, y_pred):
    print(keras.backend.sum(y_pred))
    print(keras.metrics.binary_crossentropy(y_true, y_true * y_pred))
    print((1 + keras.backend.sum(y_pred)) * keras.metrics.binary_crossentropy(y_true, y_true * y_pred))
    return (1 + keras.backend.sum(y_pred)) * keras.metrics.binary_crossentropy(y_true, y_true * y_pred)

def in_top_k_loss_single(y_true, y_pred):
    y_true_labels = tf.cast(tf.transpose(tf.where(y_true > 0))[0], tf.int32)
    y_pred = tf.reshape(y_pred, [1, tf.shape(y_pred)[0]])
    y_topk_tensor = tf.nn.top_k(y_pred, k=7)
    y_topk_ixs = y_topk_tensor[0][0][:7]
    y_topk = y_topk_tensor[1][0][:7]
    y_topk_len = tf.cast(tf.count_nonzero(y_topk_ixs), tf.int32)
    y_topk = y_topk[:y_topk_len]
    y_topk0 = tf.expand_dims(y_topk, 1)
    y_true_labels0 = tf.expand_dims(y_true_labels, 0)
    re = tf.cast(tf.reduce_any(tf.equal(y_topk0, y_true_labels0), 1), tf.int32) / tf.range(1,y_topk_len+1)
    return (-1) * tf.where(tf.equal(tf.reduce_sum(y_pred), tf.constant(0.0)), tf.constant(0.0), tf.cast(tf.reduce_mean(re),tf.float32))

### https://github.com/fchollet/keras/issues/2662
### http://stackoverflow.com/questions/37086098/does-tensorflow-map-fn-support-taking-more-than-one-tensor
# def tfmap(fn, arrays, dtype=tf.float32):
#     indices = tf.range(tf.shape(arrays[0])[0])
#     out = tf.map_fn(lambda ii: fn(*[array[ii] for array in arrays]), indices, dtype=dtype)
#     return out

def in_top_k_loss(y_true, y_pred):
    return K.mean(tf.map_fn(lambda x: in_top_k_loss_single(x[0], x[1]), (y_true, y_pred), dtype=tf.float32))
    # return K.mean(tfmap(in_top_k_loss_single, [y_true, y_pred]))


a_output_length = 24
x_output_length = 48#24
output_length = 24
input_dim = 26
attr_dim = 7
data_dim = 24#128
merged_data_dim = 12

a_input = Sequential()
a_input.add(Dense(a_output_length, activation='softmax', input_dim=attr_dim))

x_input = Sequential()
x_input.add(LSTM(data_dim, input_dim=input_dim,
    activation='sigmoid', inner_activation='hard_sigmoid', return_sequences=False))
# x_input.add(BatchNormalization()) #
# x_input.add(Activation('sigmoid')) #
x_input.add(Dropout(0.2))
x_input.add(Dense(x_output_length, activation='softmax'))

model = Sequential()
model.add(Merge([a_input, x_input], mode='concat'))
# model.add(Dense(merged_data_dim, activation='softmax'))
model.add(Dense(output_length, activation='softmax'))
# model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['mean_squared_error', 'accuracy'])
model.compile(loss=bin_crossent_true_only, optimizer=optimizers.RMSprop(lr=0.1), metrics=[in_top_k_loss, 'binary_crossentropy', 'mean_squared_error', 'accuracy'])
# model.compile(loss=in_top_k_loss, optimizer='rmsprop', metrics=[bin_crossent_true_only, 'binary_crossentropy', 'mean_squared_error', 'accuracy'])

print(time.strftime("%H:%M:%S", time.localtime()))
NUM_EPOCHS = 200
batch_size = 256
for epoch in range(NUM_EPOCHS):
    stats_all = np.zeros(len(model.metrics)+1)
    batches_count = 0
    print("epoch: " + str(epoch) + " // " + time.strftime("%H:%M:%S", time.localtime()))
    for A_train, X_train, y_train in zip(A_buckets, X_buckets, y_buckets):
        A_train, X_train, y_train = shuffle(A_train, X_train, y_train)
        A_train_batches = np.array_split(A_train, [i * batch_size for i in range(1, len(A_train) // batch_size)])
        X_train_batches = np.array_split(X_train, [i * batch_size for i in range(1, len(X_train) // batch_size)])
        y_train_batches = np.array_split(y_train, [i * batch_size for i in range(1, len(y_train) // batch_size)])
        for A_batch, X_batch, y_batch in zip(A_train_batches, X_train_batches, y_train_batches):
            stats = model.train_on_batch(x=[A_batch, X_batch], y=y_batch)
            stats_all = stats_all + stats
            batches_count += 1
        if(epoch % 10 == 9):
            print(stats)
    print(stats_all, batches_count)

print(stats)
print(time.strftime("%H:%M:%S", time.localtime()))
A_buckets = None
X_buckets = None
y_buckets = None




testdf = pd.read_csv("./testdf.csv")
df = pd.read_csv("./df.csv")
df = df.loc[df["id"].isin(testdf["id"])]
testdf = pd.concat([df, testdf], ignore_index=True, copy=False)

testdf_counts = testdf[["id", "t"]].groupby("id").count()
testdf_counts["id"] = testdf_counts.index
testdf_counts.columns = ["t_count", "id"]

testdf_maxs = df[["id", "t"]].groupby("id").max()
testdf_maxs["id"] = testdf_maxs.index
testdf_maxs.columns = ["t_max", "id"]

testdf = pd.merge(testdf_counts, testdf_maxs, how="outer", on=["id"]).merge(testdf, how="outer", on=["id"])

testdf_tmax_groups = []
for i in range(2, MAX_SEQUENCE_LENGTH + 2):
    testdf_tmax_groups.append(testdf.loc[testdf["t_count"] == i].sort_values(["id", "t"]))

testdf_attr_groups = []
for i in range(2, MAX_SEQUENCE_LENGTH + 2):
    testdf_attr_groups.append(testdf.loc[(testdf["t_count"] == i) & (testdf["t"] == 18)].sort_values(["id"])[ \
        ["t", "sex", "age", "seniority", "is_primary", "is_domestic", "income"]] \
        .fillna(value=0))

ys = df.columns.tolist()[-24:]
testdf = None

ids_test_buckets = np.array([]).reshape(0, 1)
for i in range(len(testdf_tmax_groups)):
    if not testdf_tmax_groups[i].empty:
        ids = testdf_tmax_groups[i].iloc[:, 1:2]["id"].unique()
        ids_test_buckets = np.concatenate((ids_test_buckets, ids.reshape(ids.size, 1)), axis=0)
        testdf_tmax_groups[i] = testdf_tmax_groups[i].loc[:, ['t', 't_month']+ys].as_matrix()
        testdf_tmax_groups[i] = testdf_tmax_groups[i].reshape(testdf_tmax_groups[i].shape[0]//(i+2), i+2, len(['t', 't_month']+ys))
        testdf_attr_groups[i] = testdf_attr_groups[i].as_matrix()

X_test_buckets = []
y_test_buckets = []
for g in testdf_tmax_groups:
    if g.size > 0:
        X_test_buckets.append(g[:, :-1, :])
        y_test_buckets.append(g[:, -1:, 2:].reshape(g.shape[0], g.shape[2] - 2))

testdf_tmax_groups = None
A_test_buckets = testdf_attr_groups
testdf_attr_groups = None

for a, x, y in zip(A_test_buckets, X_test_buckets, y_test_buckets):
    print(a.shape if a.size > 0 else "--", x.shape if x.size > 0 else "--", y.shape if y.size > 0 else "--")

print(ids_test_buckets.shape)




print(time.strftime("%H:%M:%S", time.localtime()))
y_test_pred = np.array([]).reshape(0, 24)
for A_test, X_test, y_test in zip(A_test_buckets, X_test_buckets, y_test_buckets):
    if A_test.size > 0 and X_test.size > 0 and y_test.size > 0:
        y_pred = model.predict([A_test, X_test], batch_size=batch_size)
        y_test_pred = np.concatenate((y_test_pred, y_pred), axis=0)

np.savetxt("./res.csv", np.concatenate((ids_test_buckets, y_test_pred), axis=1), delimiter=",")
