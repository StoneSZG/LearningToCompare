import os
import time
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages

def prelu(inp, name):
    with tf.variable_scope(name):
        i = int(inp.get_shape()[-1])
        alpha = make_var('alpha', shape=(i,))
        output = tf.nn.relu(inp) + tf.multiply(alpha, -tf.nn.relu(-inp))
    return output

def conv2d(x, shape, name, activation_func=tf.nn.relu, strides=[1, 1, 1, 1],
           trainable=True, padding='VALID', use_bn=True):
    with tf.variable_scope(name):
        w = make_var(name='weights', shape=shape, trainable=trainable,
                     initializer=tf.contrib.layers.xavier_initializer_conv2d())
        x = tf.nn.conv2d(x, w, strides=strides, padding=padding, name='conv')
        if use_bn:
            x = batch_normal(x, name='batch_normal', trainable=trainable,)
        else:
            b = make_var(name='biases', shape=shape[-1], trainable=trainable)
            x = tf.add(x, b)
        if activation_func is not None:
            x = activation_func(x, name='activate_func')

    return x

def selu(x, name):
    with tf.variable_scope(name):
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * tf.where(x >= 0.0, x, alpha * tf.nn.elu(x))

def make_var(name,
             shape,
             initializer=tf.contrib.layers.xavier_initializer(),
             dtype='float',
             collections=None,
             trainable=True):
    "A little wrapper around tf.get_variable to do weight decay and add to"
    "resnet collection"
    return tf.get_variable(name,
                           shape=shape,
                           initializer=initializer,
                           dtype=dtype,
                           regularizer=None,
                           collections=collections,
                           trainable=trainable)


def fc(inp, num_out, name, trainable=True, activation_func=None):
    # with tf.variable_scope(name):
    input_shape = inp.get_shape()
    if input_shape.ndims == 4:
        # The input is spatial. Vectorize it first.
        dim = 1
        for d in input_shape[1:].as_list():
            dim *= int(d)
        feed_in = tf.reshape(inp, [-1, dim])
    else:
        feed_in, dim = (inp, input_shape[-1].value)
    weights = make_var('weights', shape=[dim, num_out], trainable=trainable)
    biases = make_var('biases', [num_out], initializer=tf.zeros_initializer(), trainable=trainable)
    x = tf.nn.xw_plus_b(feed_in, weights, biases, name=name)
    if activation_func is not None:
        x = activation_func(x, name='activate_func')
    return x

def batch_normal(x, name, use_bias=False, trainable=True):

    MOVING_AVERAGE_DECAY = 0.9997
    BN_DECAY = MOVING_AVERAGE_DECAY
    BN_EPSILON = 0.001
    x_shape = x.get_shape()
    params_shape = x_shape[-1:]

    if use_bias:
        bias = make_var('bias', params_shape,
                        initializer=tf.zeros_initializer)
        return x + bias


    axis = list(range(len(x_shape) - 1))

    beta = make_var('beta',
                    params_shape,
                    initializer=tf.zeros_initializer)

    gamma = make_var('gamma',
                     params_shape,
                     initializer=tf.ones_initializer)

    moving_mean = make_var('moving_mean',
                           params_shape,
                           initializer=tf.zeros_initializer,
                           trainable=False)

    moving_variance = make_var('moving_variance',
                               params_shape,
                               initializer=tf.ones_initializer,
                               trainable=False)

    # These ops will only be preformed when training.
    mean, variance = tf.nn.moments(x, axis)
    update_moving_mean = moving_averages.assign_moving_average(moving_mean,
                                                               mean, BN_DECAY)

    update_moving_variance = moving_averages.assign_moving_average(
        moving_variance, variance, BN_DECAY)

    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_moving_mean)
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_moving_variance)

    mean, variance = control_flow_ops.cond(
        tf.constant(trainable, dtype=tf.bool), lambda: (mean, variance),
        lambda: (moving_mean, moving_variance))

    x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON, name=name)

    return x

def summary_variables():
    # variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    variables = tf.trainable_variables()
    for v in variables:
        # print('v:', v)

        tf.summary.histogram(v.op.name, v)
        tf.summary.scalar(v.op.name + '/avg', tf.reduce_mean(v))


def save_variables_and_metagraph(sess, saver, summary_writer, model_dir, model_name, step):
    # Save the model checkpoint
    # print('Saving variables')
    start_time = time.time()
    checkpoint_path = os.path.join(model_dir, 'model-%s.ckpt' % model_name)
    saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
    save_time_variables = time.time() - start_time
    print('Variables saved in %.2f seconds' % save_time_variables, end='')
    metagraph_filename = os.path.join(model_dir, 'model-%s.meta' % model_name)
    save_time_metagraph = 0
    if not os.path.exists(metagraph_filename):
        # print('Saving metagraph')
        start_time = time.time()
        saver.export_meta_graph(metagraph_filename)
        save_time_metagraph = time.time() - start_time
        print('Metagraph saved in %.2f seconds' % save_time_metagraph)
    summary = tf.Summary()
    #pylint: disable=maybe-no-member
    summary.value.add(tag='time/save_variables', simple_value=save_time_variables)
    summary.value.add(tag='time/save_metagraph', simple_value=save_time_metagraph)
    summary_writer.add_summary(summary, step)
