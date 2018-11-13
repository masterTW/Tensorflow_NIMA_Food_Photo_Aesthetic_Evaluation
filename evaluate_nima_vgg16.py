""" 
Implement food photo aesthetic system
Peter Tseng (twbigdata@gmail.com)
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import time
import tensorflow as tf
import tensorflow.contrib.slim as slim
slim = tf.contrib.slim
import utils
import numpy as np
from operator import itemgetter

flags = tf.app.flags
flags.DEFINE_string(
    'photo_dir', 'testimage', 'photo_dir')
flags.DEFINE_string(
    'vgg16_path', 'checkpoints/vgg16/nima-22500', 'vgg16_path')
FLAGS = tf.app.flags.FLAGS

def get_test_image(image):
    with tf.name_scope('data'):
        jpeg = tf.read_file(image)
        image = tf.image.decode_jpeg(jpeg, channels=3)
        image = utils.preprocess_image(image, is_training=False)
        image = tf.expand_dims(image, 0)
        return image

class ConvNet(object):
    def __init__(self):
        self.img = tf.placeholder(tf.float32, shape=(1, 224, 224, 3))
        self.keep_prob = tf.constant(0.75)
        self.training = False

    def get_data(self, image):
        with tf.name_scope('data'):
            jpeg = tf.read_file(image)
            image = tf.image.decode_jpeg(jpeg, channels=3)
            image = utils.preprocess_image(image, is_training=False)
            image = tf.expand_dims(image, 0)
            self.img = image

    def inference(self, weight_decay=0.0005):
        with tf.variable_scope('vgg_16', 'vgg_16', [self.img]) as sc:
            end_points_collection = sc.original_name_scope + '_end_points'
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

                        net = slim.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
                        net = slim.dropout(net, self.keep_prob, is_training=self.training,
                                         scope='dropout6')
                        net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
                        end_points = slim.utils.convert_collection_to_dict(end_points_collection)

                        net = slim.dropout(net, self.keep_prob, is_training=self.training,
                                   scope='dropout7')
                        net = slim.flatten(net)
                        net = slim.fully_connected(net, 10, scope='fc8')
                        end_points[sc.name + '/fc8'] = net
                        self.logits = net

    def expectation(self):
        '''
        Calculate the expected value
        '''
        with tf.name_scope('expectation'):
            AVA_scale = tf.transpose(tf.constant([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=tf.float32))
            self.mean = tf.matmul(self.logits, AVA_scale)

    def build(self):
        '''
        Build the computation graph
        '''
        self.inference()
        self.expectation()

    def evaluate(self):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            try:
                saver.restore(sess, FLAGS.vgg16_path)
                print('Checkpoint found')
                for (dirpath, dirnames, filenames) in os.walk(FLAGS.photo_dir):
                    final_scores = {}
                    for image_name in filenames:
                        img = get_test_image(os.path.join(dirpath, image_name))
                        score, logits = sess.run([self.mean, self.logits], feed_dict={self.img:sess.run(img)})
                        for x in score.tolist():
                            final_scores[image_name] = x[0]
                    print("(image name, aesthetic score):")
                    #sort a dictionary by value
                    print(sorted(final_scores.items(), key=itemgetter(1)))
            except Exception as e:
                print(e)


if __name__ == '__main__':
    model = ConvNet()
    model.build()
    model.evaluate()
