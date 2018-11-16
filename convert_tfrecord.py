import os
import random
import utils
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    'ava_dir', '.', 'The directory where the AVA dataset is stored.')

tf.app.flags.DEFINE_string(
    'dataset_dir', 'TFdataset', 'The directory where the TFRecord files will be stored.')

tf.app.flags.DEFINE_float(
    'test_split', 0.1, 'Fraction of dataset for validation.')

def get_File():
    ava_original_file = ['fooddrink_train.jpgl', 'fooddrink_test.jpgl']
    ava_fooddrink_file = []

    for filename in ava_original_file:
    	with open(os.path.join(FLAGS.ava_dir, 'aesthetics_image_lists', filename), 'r') as fooddrink:
    		ava_fooddrink_file.append(fooddrink.read().splitlines())
    ava_fooddrink_file = [item for sublist in ava_fooddrink_file for item in sublist]

    with open(os.path.join(FLAGS.ava_dir, 'AVA.txt'), 'r') as f:
        ava = [line.strip().split() for line in f.readlines()]
    return ava, ava_fooddrink_file

def convert_to_TFRecord(ava, ava_fooddrink_file):
    image_path = tf.placeholder(dtype=tf.string)
    jpeg = tf.read_file(image_path)
    decoded = tf.image.decode_jpeg(jpeg, channels=3)
    counts = {'train': 0, 'test': 0}
    writers = {}
    utils.safe_mkdir(FLAGS.dataset_dir)
    with tf.Session() as sess:
        for i in ava:
            for j in ava_fooddrink_file:
                    if i[1] == j:
                        filename = os.path.join(FLAGS.ava_dir, 'images', i[1]) + '.jpg'
                        try:
                            image_data, _ = sess.run(
                                [jpeg, decoded], feed_dict={image_path: filename})
                            if random.random() > FLAGS.test_split:
                                split = 'train'
                            else:
                                split = 'test'
                            if split not in writers:
                                writer_path = os.path.join(
                                    FLAGS.dataset_dir, '{}_AVA_fooddrink.tfrecord'.format(
                                        split))
                                writers[split] = tf.python_io.TFRecordWriter(writer_path)
                            scores = tf.train.FloatList(value=list(map(int, i[2:12])))
                            image = tf.train.BytesList(value=[image_data])
                            features = tf.train.Features(feature={
                                'scores': tf.train.Feature(float_list=scores),
                                'image': tf.train.Feature(bytes_list=image)})
                            example = tf.train.Example(features=features)
                            writers[split].write(example.SerializeToString())
                            counts[split] += 1
                        except:
                            print('Error decoding image: {}'.format(filename))
                            #print('Error decoding image: {},{}'.format(filename, e))

    for split, count in counts.items():
        filename = '{}.txt'.format(split)
        with open(os.path.join(FLAGS.dataset_dir, filename), 'w') as f:
            f.write('{}\n'.format(count))

def main(_):
    print("Program start")
    ava, ava_fooddrink_file = get_File()
    convert_to_TFRecord(ava, ava_fooddrink_file)
    print("end")
if __name__ == '__main__':
    tf.flags.mark_flags_as_required(['dataset_dir', 'ava_dir'])
    tf.app.run()
