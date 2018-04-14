import os
import time
import numpy as np
import tensorflow as tf
from meta_batch_iterator import MetaBatchIterator
import datasets
import tf_utils as utils

flags = tf.app.flags

flags.DEFINE_integer('hidden_units', default=8, help='')
flags.DEFINE_integer('c_way', default=5, help='class number')
flags.DEFINE_integer('k_shot', default=1, help='number of support example per class')
flags.DEFINE_integer('n_query_train', default=15, help='number of test example during a training episode')
flags.DEFINE_integer('n_query_test', default=1, help='number of test example during a test episode')
flags.DEFINE_integer('batch_size', default=1, help='number of episodes in a mini batch')
flags.DEFINE_integer('n_rotations', default=4, help='number of rotations to consider to augment number of classes (min=1, max=4)')
flags.DEFINE_float('learning_rate', default=1e-3, help='Initial learning rate.')
flags.DEFINE_integer('decay_steps', default=1000, help='learning rate decay steps')
flags.DEFINE_integer('epoches', default=1000000, help='learning rate decay steps')
flags.DEFINE_float('decay_rate', default=0.99, help='learning rate decay rate')
flags.DEFINE_float('max_gradient_norm', default=0.5, help='lips values of multiple tensors by the ratio of the sum of their norms')
flags.DEFINE_string('omniglot_path', default='omniglot', help='Directory containing the omniglot dataset.')
flags.DEFINE_integer('n_train_classes', default=1200, help='number of classes for training (without considering rotations) (omniglot has 1623 classes)')
flags.DEFINE_integer('test_interval', default=100, help='number of steps before testing')
flags.DEFINE_integer('n_test_episodes', default=10, help='number of steps before testing')
flags.DEFINE_string('log_dir', default='./logs', help='logs path')
flags.DEFINE_string('model_dir', default='./models', help='save model path')

args = flags.FLAGS

def CNNEncoder(image, trainable=True):
    x = image
    with tf.variable_scope('CNNEncoder', initializer=tf.contrib.layers.xavier_initializer(),
                           ) :
        with tf.variable_scope('layer1'):
            x = utils.conv2d(x, name='conv1', shape=[3, 3, 1, 64], padding='SAME',
                             activation_func=tf.nn.relu,
                             trainable=trainable,
                             use_bn=True)
            x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME')

        with tf.variable_scope('layer2'):
            x = utils.conv2d(x, name='conv1', shape=[3, 3, 64, 64],
                             padding='SAME',
                             activation_func=tf.nn.relu,
                             trainable=trainable,
                             use_bn=True)
            x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME')

        with tf.variable_scope('layer3'):
            x = utils.conv2d(x, name='conv1', shape=[3, 3, 64, 64],
                             padding='SAME',
                             activation_func=tf.nn.relu,
                             trainable=trainable,
                             use_bn=True)
            x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME')

        with tf.variable_scope('layer4'):
            x = utils.conv2d(x, name='conv1', shape=[3, 3, 64, 64],
                             padding='SAME',
                             activation_func=tf.nn.relu,
                             trainable=trainable,
                             use_bn=True)

    return x

def RelationNetwork(encoder, hidden_size, trainable=True):
    x = encoder
    with tf.variable_scope('RelationNetwork') as scope:
        with tf.variable_scope('layer1'):
            x = utils.conv2d(x, name='conv1',
                             shape=[3, 3, 128, 64],
                             padding='SAME',
                             activation_func=tf.nn.relu,
                             trainable=trainable,
                             use_bn=True)
            x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        with tf.variable_scope('layer2'):
            x = utils.conv2d(x, name='conv1',
                             shape=[3, 3, 64, 64],
                             padding='SAME',
                             activation_func=tf.nn.relu,
                             trainable=trainable,
                             use_bn=True)
            x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        with tf.variable_scope('fc1'):
            x = utils.fc(x, num_out=hidden_size, name='fc1', activation_func=tf.nn.relu)

        with tf.variable_scope('fc2'):
            x = utils.fc(x, num_out=1, name='fc2', activation_func=None)

    return x

def main(_):
    if not os.path.isdir(args.log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(args.log_dir)

    if not os.path.isdir(args.model_dir):  # Create the model directory if it doesn't exist
        os.makedirs(args.model_dir)

    # Step 1: init data folders
    print("init data")
    example_size = (28, 28, 1)
    rotations = list(range(args.n_rotations))
    # init character folders for dataset construction
    omniglot = datasets.Omniglot(root=args.omniglot_path, download=True, rotations=rotations,
                                 split=args.n_train_classes, example_size=example_size)

    train_batch_iterator = MetaBatchIterator(omniglot.train, args.batch_size, args.c_way, args.k_shot, args.n_query_train)
    test_batch_iterator = MetaBatchIterator(omniglot.test, args.batch_size, args.c_way, args.k_shot, args.n_query_test)

    supports, targets, targets_labels = train_batch_iterator.get_placeholders()

    supports_set = tf.reshape(supports, [-1, *example_size])
    targets_set = tf.reshape(targets, [-1, *example_size])
    labels = tf.reduce_mean(targets_labels, axis=3)
    targets_set_labels = tf.reshape(labels, [-1, 1])

    with tf.variable_scope('Encoder') as scope:
        supports_set_encoder = CNNEncoder(supports_set)
        scope.reuse_variables()
        targets_set_encoder = CNNEncoder(targets_set, trainable=True)

    supports_set_encoder = tf.reshape(supports_set_encoder, (args.batch_size, -1, args.c_way,
                                                             args.k_shot, 4, 4, 64))
    supports_set_encoder = tf.reduce_sum(supports_set_encoder, axis=3)
    supports_set_encoder = tf.reshape(supports_set_encoder, [-1, 4, 4, 64])

    targets_set_encoder = tf.reshape(targets_set_encoder, (args.batch_size, -1, args.c_way,
                                                             args.k_shot, 4, 4, 64))
    targets_set_encoder = tf.reduce_sum(targets_set_encoder, axis=3)
    targets_set_encoder = tf.reshape(targets_set_encoder, [-1, 4, 4, 64])

    relation_pairs = tf.concat([supports_set_encoder, targets_set_encoder], axis=-1)
    # print('relation_pairs:', relation_pairs)

    relations = RelationNetwork(relation_pairs, args.hidden_units)
    # relations = tf.reshape(relations, (-1, args.class_num))
    relations_normalized = tf.nn.sigmoid(relations)
    # print('relations:', relations)

    losses = tf.reduce_mean(tf.squared_difference(relations_normalized, targets_set_labels))
    tf.summary.scalar('losses', losses)

    y_test = tf.reshape(labels, [args.batch_size, -1, args.c_way])
    y_correct_idx = tf.argmax(y_test, axis=2)

    output_test = tf.reshape(relations, [args.batch_size, -1, args.c_way])
    max_idx = tf.argmax(output_test, axis=2)
    correct_prediction = tf.equal(y_correct_idx, max_idx)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
    tf.summary.scalar('accuracy', accuracy)

    global_step = tf.train.get_or_create_global_step()
    tf.summary.scalar('global_step', global_step)

    learning_rate = tf.train.exponential_decay(args.learning_rate, global_step=global_step,
                                               decay_steps=args.decay_steps,
                                               decay_rate=args.decay_rate)
    tf.summary.scalar('learning_rate', learning_rate)

    utils.summary_variables()

    params = tf.trainable_variables()

    optimizer = tf.train.AdamOptimizer(learning_rate)
    clipped_gradients, norm = tf.clip_by_global_norm(tf.gradients(losses, params),
                                                     args.max_gradient_norm)
    train_op = optimizer.apply_gradients(zip(clipped_gradients, params), global_step=global_step)

    # Create a saver
    saver = tf.train.Saver(max_to_keep=5)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter(args.log_dir, sess.graph)
        start_time = time.time()
        for i in range(args.epoches):
            support_images, target_images, target_labels = train_batch_iterator.get_inputs()

            feed_dict = {supports: support_images,
                         targets: target_images,
                         targets_labels: target_labels}

            cost, acc, _ = sess.run([losses, accuracy, train_op], feed_dict=feed_dict)

            if (i % args.test_interval) == 0:
                total_accuracies = []
                total_losses = []
                print('(s: %d, e: %d) train: loss, acc : %.4f, %.4f' % (i, i * args.batch_size, cost, acc))
                for j in range(args.n_test_episodes):
                    support_images, target_images, target_labels = test_batch_iterator.get_inputs()

                    feed_dict = {supports: support_images,
                                targets: target_images,
                                targets_labels: target_labels}
                    cost, acc, = sess.run([losses, accuracy], feed_dict=feed_dict)

                    total_accuracies.append(acc)
                    total_losses.append(cost)
                mean_acc = np.mean(total_accuracies)
                std_acc = np.std(total_accuracies)
                mean_loss = np.mean(total_losses)
                duration = time.time() - start_time
                print('test(%.2fs): loss, acc (%d es): %.4f, %.4f(%.2f)' % (duration, args.n_test_episodes, mean_loss,
                                                                              mean_acc, std_acc))
                summary, step = sess.run([summary_op, global_step], feed_dict=feed_dict)
                summary_writer.add_summary(summary, step)
                utils.save_variables_and_metagraph(sess, saver, summary_writer,
                                                   args.model_dir, 'LearningToCompare', step)

if __name__ == '__main__':
    tf.app.run()
