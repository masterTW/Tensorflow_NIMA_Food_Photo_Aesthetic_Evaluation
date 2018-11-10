import os
import gzip
import shutil
import struct
import urllib

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import tensorflow as tf

IMAGE_SIZE = 224

def safe_mkdir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass

def preprocess_image(image, height=IMAGE_SIZE, width=IMAGE_SIZE,
                     rescale_height=256, rescale_width=256,
                     central_fraction=0.875, is_training=True, scope=None):
    """Pre-process a batch of images for training or evaluation.
    Args:
      image: a tensor of shape [height, width, channels] with the image.
      height: optional Integer, image expected height.
      width: optional Integer, image expected width.
      rescale_height: optional Integer, rescaling height before cropping.
      rescale_width: optional Integer, rescaling width before cropping.
      central_fraction: optional Float, fraction of the image to crop.
      is_training: if true it would transform an image for training,
        otherwise it would transform it for evaluation.
      scope: optional name scope.
    Returns:
      3-D float Tensor containing a preprocessed image.
    """

    with tf.name_scope(scope, 'prep_image', [image, height, width]):
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)

        if is_training:
            image = tf.image.resize_images(
                image, [rescale_height, rescale_width])
            image = tf.random_crop(image, size=(height, width, 3))
            image = tf.image.random_flip_left_right(image)
        else:
            image = tf.image.resize_images(image, [height, width])

        tf.summary.image('final_sampled_image', tf.expand_dims(image, 0))
        image = tf.subtract(image, 0.5)
        image = tf.multiply(image, 2.0)
        return image

def preprocess(example, num_classes=10, is_training=True):
    """Extract and preprocess dataset features.
    Args:
      example: an instance of protobuf-encoded example.
      num_classes: number of predicted classes. Defaults to 10.
      is_training: whether is training or not.
    Returns:
      A tuple of `image` and `scores` tensors.
    """
    features = {'scores': tf.VarLenFeature(tf.float32),
                'image': tf.FixedLenFeature((), tf.string)}
    parsed = tf.parse_single_example(example, features)
    image = tf.image.decode_jpeg(parsed['image'], channels=3)
    image = preprocess_image(image, is_training=is_training)
    scores = parsed['scores']
    scores = tf.sparse_tensor_to_dense(scores)
    scores = tf.reshape(scores, [num_classes])
    scores = scores / tf.reduce_sum(scores, axis=-1, keepdims=True)
    return image, scores

def preprocess_testdata(example, num_classes=10, is_training=False):
    """Extract and preprocess dataset features.
    Args:
      example: an instance of protobuf-encoded example.
      num_classes: number of predicted classes. Defaults to 10.
      is_training: whether is training or not.
    Returns:
      A tuple of `image` and `scores` tensors.
    """
    features = {'scores': tf.VarLenFeature(tf.float32),
                'image': tf.FixedLenFeature((), tf.string)}
    parsed = tf.parse_single_example(example, features)
    image = tf.image.decode_jpeg(parsed['image'], channels=3)
    image = preprocess_image(image, is_training=is_training)
    scores = parsed['scores']
    scores = tf.sparse_tensor_to_dense(scores)
    scores = tf.reshape(scores, [num_classes])
    scores = scores / tf.reduce_sum(scores, axis=-1, keepdims=True)
    return image, scores

def get_nima_dataset(batch_size):

    train_folder = os.path.join('TFdataset', '{}_*.tfrecord'.format('train'))
    train_file = tf.data.Dataset.list_files(train_folder)
    train_dataset = tf.data.TFRecordDataset(train_file)
    train_dataset = train_dataset.shuffle(1000)
    #dataset = dataset.repeat()
    train_dataset = train_dataset.map(preprocess)
    train_dataset = train_dataset.batch(batch_size)

    test_folder = os.path.join('TFdataset', '{}_*.tfrecord'.format('test'))
    test_file = tf.data.Dataset.list_files(test_folder)
    test_dataset = tf.data.TFRecordDataset(test_file)
    #dataset = dataset.repeat()
    test_dataset = test_dataset.map(preprocess_testdata)
    test_dataset = test_dataset.batch(batch_size)
    return train_dataset, test_dataset

def tril_indices(n, k=0):
    """Return the indices for the lower-triangle of an (n, m) array.
    Works similarly to `np.tril_indices`
    Args:
      n: the row dimension of the arrays for which the returned indices will
        be valid.
      k: optional diagonal offset (see `np.tril` for details).
    Returns:
      inds: The indices for the triangle. The returned tuple contains two arrays,
        each with the indices along one dimension of the array.
    """
    m1 = tf.tile(tf.expand_dims(tf.range(n), axis=0), [n, 1])
    m2 = tf.tile(tf.expand_dims(tf.range(n), axis=1), [1, n])
    mask = (m1 - m2) >= -k
    ix1 = tf.boolean_mask(m2, tf.transpose(mask))
    ix2 = tf.boolean_mask(m1, tf.transpose(mask))
    return ix1, ix2

def ecdf(p):
    """Estimate the cumulative distribution function.
    The e.c.d.f. (empirical cumulative distribution function) F_n is a step
    function with jump 1/n at each observation (possibly with multiple jumps
    at one place if there are ties).
    For observations x= (x_1, x_2, ... x_n), F_n is the fraction of
    observations less or equal to t, i.e.,
    F_n(t) = #{x_i <= t} / n = 1/n \sum^{N}_{i=1} Indicator(x_i <= t).
    Args:
      p: a 2-D `Tensor` of observations of shape [batch_size, num_classes].
        Classes are assumed to be ordered.
    Returns:
      A 2-D `Tensor` of estimated ECDFs.
    """
    n = p.get_shape().as_list()[1]
    indices = tril_indices(n)
    indices = tf.transpose(tf.stack([indices[1], indices[0]]))
    ones = tf.ones([n * (n + 1) / 2])
    triang = tf.scatter_nd(indices, ones, [n, n])
    return tf.matmul(p, triang)
