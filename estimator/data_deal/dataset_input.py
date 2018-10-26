import os
from functools import partial
import tensorflow as tf
from tensorflow.python.ops.image_ops_impl import ResizeMethod


# 函数的功能时将filename对应的图片文件读进来，并缩放到统一的大小，再normalize
def load_image_tf(filename, label, height, width, channels=3):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_image(image_string, channels)
    image_decoded.set_shape([None, None, None])
    image_decoded = tf.image.central_crop(image_decoded, 1)
    image_decoded = tf.image.resize_images(image_decoded, tf.constant([height, width], tf.int32),
    method=ResizeMethod.NEAREST_NEIGHBOR)
    image_resized = tf.image.resize_image_with_crop_or_pad(image_decoded, height, width)
    image_resized = tf.reshape(image_resized, [height, width, channels])
    image_resized = tf.divide(image_resized, 255)
    image_resized = tf.subtract(image_resized, 0.5)
    image_resized = tf.multiply(image_resized, 2.)
    return image_resized, label

def read_folder(folders, labels):
    if not isinstance(folders, (list, tuple, set)):
        raise ValueError("folders 应为list 或 tuple")
    all_files = []
    all_labels = []
    for i, f in enumerate(folders):
        files = os.listdir(f)
        for file in files:
            all_files.append(os.path.join(f, file))
            all_labels.append(labels[i])
    dataset = tf.data.Dataset.from_tensor_slices((all_files, all_labels))
    return dataset, len(all_files)


def dataset_input_fn(folders, labels, epoch, batch_size,
                    height, width, channels,
                    scope_name="dataset_input",
                    feature_name=None):
    def fn():
        with tf.name_scope(scope_name):
            dataset, l = read_folder(folders, labels)
            dataset = dataset.map(partial(load_image_tf, height=height, width=width, channels=channels))
            dataset = dataset.shuffle(buffer_size=l).repeat(epoch).batch(batch_size)
            iterator = dataset.make_one_shot_iterator()
            one_element = iterator.get_next()
            if feature_name:
                return {str(feature_name): one_element[0]}, one_element[1]
            return one_element[0], one_element[1]
    return fn