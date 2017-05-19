import keras
import tensorflow as tf
import keras.backend as K


def bin_crossentropy_true_only(y_true, y_pred):
    return (1 + K.sum(y_pred, axis=-1)) * K.mean(K.binary_crossentropy(y_true * y_pred, y_true), axis=-1)


# def tf_sparse_precision_at_k(y_true, y_pred):
#     y_true_labels = tf.transpose(tf.where(y_true > 0))[0]
#     return tf.metrics.sparse_precision_at_k(y_true_labels, y_pred, 3)[0]


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


def in_top_k_loss(y_true, y_pred):
    return K.mean(tf.map_fn(lambda x: in_top_k_loss_single(x[0], x[1]), (y_true, y_pred), dtype=tf.float32))

