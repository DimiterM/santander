import time
import pandas as pd
import numpy as np

import dataset
from functions import *

import keras.backend as K
from keras.models import load_model

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt



def get_merged_layer_function(model):
    return K.function([model.layers[1].input, model.layers[0].input, K.learning_phase()], [model.layers[-5].output])

def get_layer_activations(A, X, f_layer):
    outf = tuple()
    b = 100000
    i = 1
    while (i-1)*b <= X_train.shape[0]:
        outf = outf + (  f_layer([A_train[(i-1)*b:i*b], X_train[(i-1)*b:i*b], 0])[0]  ,  )
        i += 1
    return np.vstack(outf)


# code from https://medium.com/@awjuliani/visualizing-deep-learning-with-t-sne-tutorial-and-video-e7c59ee4080c
def plot_with_labels(lowDWeights, labels, filename="./tsne/tsne.png"):
    assert lowDWeights.shape[0] >= len(labels), "More labels than weights"
    plt.figure(figsize=(20, 20))  #in inches
    for i, label in enumerate(labels):
        x, y = lowDWeights[i,:]
        plt.scatter(x, y, c=('g' if label else 'r'))
        #plt.annotate(label,
        #         xy=(x, y),
        #         xytext=(5, 2),
        #         textcoords='offset points',
        #         ha='right',
        #         va='bottom')
    plt.savefig(filename)
    plt.clf()




print(time.strftime("%H:%M:%S", time.localtime()))
train_month = 16
# attr_cols = ["t", "sex", "age", "seniority", "is_primary", "is_domestic", "income"]
attr_cols = ['t', 't_month', 'sex', 'age', 'seniority_new', 'seniority', 'is_primary', 'is_domestic', 'is_foreigner', 'is_dead', 'is_active', 'income', 'employee_bit_notN', 'country_num', 'customer_type_bit_not1', 'customer_rel_I', 'customer_rel_A', 'channel_0_0', 'channel_0_1', 'channel_0_2', 'channel_1_0', 'channel_1_1', 'channel_1_2', 'channel_1_3', 'channel_1_4', 'channel_1_5', 'channel_1_6', 'channel_1_7', 'channel_1_8', 'segment_1', 'segment_2', 'segment_3']
remove_non_buyers = False
scale_time_dim = False
include_time_dim_in_X = True

A_train, X_train, y_train = dataset.load_padded_trainset(max_month=train_month, attr_cols=attr_cols, 
    remove_non_buyers=remove_non_buyers, scale_time_dim=scale_time_dim, include_time_dim_in_X=include_time_dim_in_X)

print(A_train.shape, X_train.shape, y_train.shape)


print(time.strftime("%H:%M:%S", time.localtime()))
model_filename = "./models/model_val_06-14_05-09.h5"
custom_objects = {"bin_crossentropy_true_only": bin_crossentropy_true_only, "in_top_k_loss": in_top_k_loss}
model = load_model(model_filename, custom_objects=custom_objects)


### get activations
print(time.strftime("%H:%M:%S", time.localtime()))
print("get activations")
activations = get_layer_activations(A_train, X_train, get_merged_layer_function(model))


### plot embeddings
print(time.strftime("%H:%M:%S", time.localtime()))
print("plot embeddings")

for c in range(dataset.NUM_CLASSES):
    s = StratifiedShuffleSplit(test_size=0.005)
    _, i = next( s.split(activations, y_train[:,c]) )
    X_acts, y = activations[i], y_train[i][:,c]
    print(c+1, y[y > 0].shape[0], int(y_train[:,c].sum()))
    
    t = TSNE(n_components=2, perplexity=30.0, init='pca', n_iter=200)#1000)
    tsne_activations = t.fit_transform(X_acts)
    plot_with_labels(tsne_activations, y, "./tsne/tsne_"+str(c+1)+".png")


print(time.strftime("%H:%M:%S", time.localtime()))

###############################################################


# # select ids with max activation values
# top_n_samples = 4
# for i in range(activations.shape[1]):
#     ind = np.argpartition(activations[:,i], -top_n_samples)[-top_n_samples:]
#     ind = ind[np.argsort(activations[:,i][ind])]
#     print(i, ind, activations[:,i][ind][::-1])
# #


A_train, X_train, y_train = None, None, None
print(time.strftime("%H:%M:%S", time.localtime()))
test_month = 17
A_test, X_test, y_test, ids_test = dataset.load_padded_testset(train_month=train_month, test_month=test_month, attr_cols=attr_cols, 
    remove_non_buyers=remove_non_buyers, scale_time_dim=scale_time_dim, include_time_dim_in_X=include_time_dim_in_X)

print(A_test.shape, X_test.shape, y_test.shape, ids_test.shape)
print(time.strftime("%H:%M:%S", time.localtime()))

activations = get_layer_activations(A_test, X_test, get_merged_layer_function(model))

### select ids with max activation values for each node in merge layer
print(time.strftime("%H:%M:%S", time.localtime()))
print("get examples with max activation values (from testset)")
top_n_samples = 10
for i in range(activations.shape[1]):
    ind = np.argpartition(activations[:,i], -top_n_samples)[-top_n_samples:]
    ind = ind[np.argsort(activations[:,i][ind])]
    # print(i, ind, activations[:,i][ind][::-1])
    print("Class "+str(i+1))
    print(ids_test[ind].tolist())


print(time.strftime("%H:%M:%S", time.localtime()))

"""
model.summary()
f = K.function([model.layers[1].input, model.layers[0].input, K.learning_phase()], [model.layers[-5].output])
outf = f([A_test, X_test, 0])[0]
"""

