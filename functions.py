import keras
import tensorflow as tf
import keras.backend as K

K_VAL = 7


def bin_crossentropy_true_only(y_true, y_pred):
    return (1 + K.sum(y_pred, axis=-1)) * K.mean(K.binary_crossentropy(y_true * y_pred, y_true), axis=-1)


# def tf_sparse_precision_at_k(y_true, y_pred):
#     y_true_labels = tf.transpose(tf.where(y_true > 0))[0]
#     return tf.metrics.sparse_precision_at_k(y_true_labels, y_pred, K_VAL)[0]


def in_top_k_loss_single(y_true, y_pred):
    y_true_labels = tf.cast(tf.transpose(tf.where(y_true > 0))[0], tf.int32)
    y_topk_tensor = tf.nn.top_k(y_pred, k=K_VAL)
    y_topk_ixs = y_topk_tensor[0][:K_VAL]
    y_topk = y_topk_tensor[1][:K_VAL]
    y_true_len = tf.cast(tf.count_nonzero(y_true), tf.int32)
    y_topk0 = tf.expand_dims(y_topk, 1)
    y_true_labels0 = tf.expand_dims(y_true_labels, 0)
    hits = tf.cast(tf.reduce_any(tf.equal(y_topk0, y_true_labels0), 1), tf.int32)
    hit_counts_at_k = tf.cumsum(hits) * hits
    prec_on_hits = hit_counts_at_k / tf.range(1,K_VAL+1)
    mapk = tf.cast(tf.reduce_sum(prec_on_hits) / tf.cast(tf.minimum(y_true_len, K_VAL), tf.float64), tf.float32)
    return (-1) * tf.where(tf.equal(tf.cast(y_true_len, tf.float32), tf.constant(0.0)), tf.constant(0.0), mapk)


def in_top_k_loss(y_true, y_pred):
    return K.mean(tf.map_fn(lambda x: in_top_k_loss_single(x[0], x[1]), (y_true, y_pred), dtype=tf.float32))


def in_top_k_loss_new_only(y_true, y_pred, y_last):
    y_true_new_only = tf.clip_by_value(y_true - y_last, 0.0, 1.0)
    y_pred_new_only = (tf.ones(y_true.shape) - y_last) * y_pred
    return K.mean(tf.map_fn(lambda x: in_top_k_loss_single(x[0], x[1]), (y_true_new_only, y_pred_new_only), dtype=tf.float32))



def calculate_top_k_new_only(model, A_test, X_test, y_test, batch_size, include_time_dim_in_X):
    y_pred = model.predict([A_test, X_test], batch_size) if isinstance(model, keras.engine.training.Model) else model.predict(A_test, X_test, batch_size)
    return (-1) * K.eval(
        in_top_k_loss_new_only(
            y_test, y_pred, 
            X_test[:,-1:,(2 if include_time_dim_in_X else 0):].reshape(X_test.shape[0], X_test.shape[2] - (2 if include_time_dim_in_X else 0)))
        )

def calculate_top_k_new_only_maxdataset(model, X_test, y_test, batch_size):
    y_pred = model.predict(X_test, batch_size)
    return (-1) * K.eval(
        in_top_k_loss_new_only(
            y_test, y_pred, 
            X_test[:,-48:-24])
        )

def calculate_top_k_new_only_concatdataset(model, X_test, y_test, batch_size, len_attr_cols):
    y_pred = model.predict(X_test, batch_size)
    return (-1) * K.eval(
        in_top_k_loss_new_only(
            y_test, y_pred, 
            X_test[:,len_attr_cols:len_attr_cols+24])
        )

