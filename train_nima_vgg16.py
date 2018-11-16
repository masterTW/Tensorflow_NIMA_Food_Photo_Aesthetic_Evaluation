""" 
Food photo aesthetic scoring system
Peter Tseng (twbigdata@gmail.com)
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import time
import tensorflow as tf
import tensorflow.contrib.slim as slim
slim = tf.contrib.slim
import utils
import logging

logging.basicConfig(filename='logger.log',level=logging.INFO)
flags = tf.app.flags
tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')
flags.DEFINE_string(
    'log_level', 'info', 'log level')
flags.DEFINE_integer(
    'n_epochs', '100', 'number of epoch')
flags.DEFINE_integer(
    'batch_size', '20', 'batch_size')
flags.DEFINE_float(
    'learning_rate', '1e-6', 'learning rate')
tf.app.flags.DEFINE_string(
    'checkpoint_exclude_scopes', "vgg_16/fc8",
    'Comma-separated list of scopes of variables to exclude when restoring '
    'from a checkpoint.')
tf.app.flags.DEFINE_string(
    'checkpoint_path', 'pretrained/vgg16/vgg_16.ckpt', 'Path to the model checkpoint.')
FLAGS = tf.app.flags.FLAGS

def _get_init_fn():
    """Return a function that 'warm-starts' the training.
    Returns:
      An init function.
    """
    exclusions = []
    if FLAGS.checkpoint_exclude_scopes:
        exclusions = [scope.strip()
                      for scope in FLAGS.checkpoint_exclude_scopes.split(',')]
    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)
    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
        checkpoint_path = FLAGS.checkpoint_path
    tf.logging.info('Fine-tuning from {}'.format(checkpoint_path))

    return slim.assign_from_checkpoint_fn(checkpoint_path, variables_to_restore)

class ConvNet(object):
    def __init__(self):
        self.lr = FLAGS.learning_rate
        self.batch_size = FLAGS.batch_size
        self.keep_prob = tf.constant(0.75)
        self.gstep = tf.Variable(0, dtype=tf.int64, 
                                trainable=False, name='global_step')
        self.batches_epoch = 5000//FLAGS.batch_size
        self.skip_step = 20
        self.training = False
        print("Start")
        print("learning_rate:{}".format(self.lr))
        print("batch_size:{}".format(self.batch_size))
        print("n_epochs:{}".format(FLAGS.n_epochs))

    def get_data(self):
        with tf.name_scope('data'):
            train_data, test_data = utils.get_nima_dataset(FLAGS.dataset_dir, self.batch_size)
            iterator = tf.data.Iterator.from_structure(train_data.output_types, 
                                                   train_data.output_shapes)
            self.img, self.label = iterator.get_next()
            self.train_init = iterator.make_initializer(train_data)  # initializer for train_data
            self.test_init = iterator.make_initializer(test_data)    # initializer for train_data

    def inference(self, weight_decay=0.0005):
        with tf.variable_scope('vgg_16', 'vgg_16', [self.img]) as sc:
            end_points_collection = sc.original_name_scope + '_end_points'
            # Collect outputs for conv2d, fully_connected and max_pool2d.
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_regularizer=slim.l2_regularizer(weight_decay),
                      biases_initializer=tf.zeros_initializer()):
                with slim.arg_scope([slim.conv2d], padding='SAME') :
                    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                                  outputs_collections=end_points_collection):
                        net = slim.repeat(self.img, 2, slim.conv2d, 64, [3, 3], scope='conv1')
                        net = slim.max_pool2d(net, [2, 2], scope='pool1')
                        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
                        net = slim.max_pool2d(net, [2, 2], scope='pool2')
                        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
                        net = slim.max_pool2d(net, [2, 2], scope='pool3')
                        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
                        net = slim.max_pool2d(net, [2, 2], scope='pool4')
                        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
                        net = slim.max_pool2d(net, [2, 2], scope='pool5')

                        # Use conv2d instead of fully_connected layers.
                        net = slim.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
                        net = slim.dropout(net, self.keep_prob, is_training=self.training,
                                         scope='dropout6')
                        net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
                        # Convert end_points_collection into a end_point dict.
                        end_points = slim.utils.convert_collection_to_dict(end_points_collection)

                        net = slim.dropout(net, self.keep_prob, is_training=self.training,
                                   scope='dropout7')
                        net = slim.flatten(net)
                        net = slim.fully_connected(net, 10, scope='fc8')
                        end_points[sc.name + '/fc8'] = net
                        self.logits = net

    def loss(self):
        """
        Hou, Le, Chen-Ping Yu, and Dimitris Samaras. "Squared Earth Mover's
        Distance-based Loss for Training Deep Neural Networks." arXiv preprint
        arXiv:1611.05916 (2016).
        """
        with tf.name_scope(None, 'EmdLoss', [self.label, self.logits]):
            ecdf_p = utils.ecdf(self.label)
            ecdf_p_hat = utils.ecdf(self.logits)
            emd = tf.reduce_mean(tf.pow(tf.abs(ecdf_p - ecdf_p_hat), 2), axis=-1)
            emd = tf.pow(emd, 1 / 2)
            self.loss = tf.reduce_mean(emd)

    def optimize(self):
        '''
        Define training op
        using GradientDescentOptimizer to minimize cost
        '''
        learning_rate = tf.train.exponential_decay(
                self.lr, self.gstep, self.batches_epoch,
                0.95, staircase=True,
                name='exponential_decay_learning_rate')
        full_optimizer = tf.train.AdamOptimizer(learning_rate)
        self.opt = full_optimizer.minimize(self.loss)



    def summary(self):
        '''
        Create summaries to write on TensorBoard
        '''
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
            self.summary_op = tf.summary.merge_all()


    def expectation(self):
        '''
        Calculate the expected value
        '''
        with tf.name_scope('expectation'):
            AVA_scale = tf.transpose(tf.constant([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=tf.float32))
            self.mean = tf.matmul(self.logits, AVA_scale)
            self.mean_gt = tf.matmul(self.label, AVA_scale)

    def build(self):
        '''
        Build the computation graph
        '''
        self.get_data()
        self.inference()
        self.loss()
        self.optimize()
        self.summary()
        self.expectation()

    def train_one_epoch(self, sess, saver, init, writer, epoch, step):
        start_time = time.time()
        sess.run(init)
        self.training = True
        total_loss = 0
        n_batches = 0
        total_correct_preds = 0
        train_dataset_sum = 0
        try:
            while True:
                score, score_gt, _, l, summaries = sess.run([self.mean, self.mean_gt, self.opt, self.loss, self.summary_op])
                writer.add_summary(summaries, global_step=step)
                if (step + 1) % self.skip_step == 0:
                    print('Loss at step {0}: {1}'.format(step, l))
                step += 1
                total_loss += l
                n_batches += 1
                score_list = []
                scoregt_list = []
                for x in score.tolist():
                    if x[0] >=5:
                        score_list.append(1)
                    else:
                        score_list.append(0)

                for x in score_gt.tolist():
                    if x[0] >=5:
                        scoregt_list.append(1)
                    else:
                        scoregt_list.append(0)
                if len(score_list)==len(scoregt_list):
                    for id, x in enumerate(score_list):
                        if x == scoregt_list[id]:
                            total_correct_preds += 1
                            train_dataset_sum += 1
                        else:
                            train_dataset_sum += 1
        except tf.errors.OutOfRangeError:
            pass
        print('Accuracy at epoch {0}: {1} '.format(epoch, total_correct_preds/train_dataset_sum))
        logging.warning('Train accuracy at epoch {0}: {1} '.format(epoch, total_correct_preds/train_dataset_sum))
        saver.save(sess, 'checkpoints/vgg16/nima', step)
        print('Average loss at epoch {0}: {1}'.format(epoch, total_loss/n_batches))
        print('Took: {0} seconds'.format(time.time() - start_time))
        return step

    def eval_once(self, sess, init, writer, epoch, step):
        start_time = time.time()
        sess.run(init)
        self.training = False
        total_correct_preds = 0
        testdataset_sum = 0
        try:
            while True:
                score, score_gt = sess.run([self.mean, self.mean_gt])
                score_list = []
                scoregt_list = []
                for x in score.tolist():
                    if x[0] >=5:
                        score_list.append(1)
                    else:
                        score_list.append(0)
                for x in score_gt.tolist():
                    if x[0] >=5:
                        scoregt_list.append(1)
                    else:
                        scoregt_list.append(0)
                if len(score_list)==len(scoregt_list):
                    #print(total_correct_preds)
                    for id, x in enumerate(score_list):
                        if x == scoregt_list[id]:
                            total_correct_preds += 1
                            testdataset_sum += 1
                        else:
                            testdataset_sum += 1
        except tf.errors.OutOfRangeError as e:
            pass

        print('Accuracy at epoch {0}: {1} '.format(epoch, total_correct_preds/testdataset_sum))
        logging.warning('Test accuracy at epoch {0}: {1} '.format(epoch, total_correct_preds/testdataset_sum))
        print('Took: {0} seconds'.format(time.time() - start_time))

    def train(self, n_epochs):
        '''
        The train function alternates between training one epoch and evaluating
        '''
        print("Program start")
        utils.safe_mkdir('checkpoints')
        utils.safe_mkdir('checkpoints/vgg16')
        writer = tf.summary.FileWriter('./graphs/vgg16', tf.get_default_graph())
        init_fn=_get_init_fn()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            init_fn(sess)
            saver = tf.train.Saver()
            step = self.gstep.eval()
            for epoch in range(n_epochs):
                step = self.train_one_epoch(sess, saver, self.train_init, writer, epoch, step)
                self.eval_once(sess, self.test_init, writer, epoch, step)
        writer.close()

if __name__ == '__main__':
    model = ConvNet()
    model.build()
    model.train(n_epochs=FLAGS.n_epochs)
