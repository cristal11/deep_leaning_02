import time
import matplotlib.pyplot as plt
import tensorflow as tf
import imageio
import numpy as np
import os
import data_prepare.picture_preprocess as pre_process

from functools import partial

TFRECORD_FLIE = ""

HEIGHT = 299
WEIGHT = 299
CHANNEL = 3

BATCH_SIZE = 1
SHUFFLE_BIFFER = 360
NUM_EPOCH = 1


# 解析
def _parse(record, height, width, channel=3):
    features = tf.parse_single_example(
        record,
        features={
            "image": tf.FixedLenFeature([], tf.string),
            "label": tf.FixedLenFeature([], tf.int64)
        }
    )
    decoded_image = tf.image.decode_jpeg(features["image"], channels=channel)
    if decoded_image.dtype != tf.float32:
        decoded_image = tf.image.convert_image_dtype(decoded_image, tf.float32)
    image = tf.image.resize_images(decoded_image, (height, width))
    image.set_shape([height, width, channel])

    return image, features["label"]


def gen_dataset_input_fn(file_pattern, height, width, channel=3,
                         shuffle_biffer=360, batch_size=36,
                         num_epoch=1, features_name=None,
                         ):
    def fn():
        # 获取tfrecord文件
        files = tf.train.match_filenames_once(file_pattern)
        # 生成数据集
        dataset = tf.data.TFRecordDataset(files)
        # 解析tfrecord,将二进制转换为tensor
        dataset = dataset.map(lambda record: (_parse(record, height, width, channel)))
        # 图片预处理
        dataset = dataset.map(
            lambda image, label: (pre_process.preprocess_for_train(image, height, width, None), label)
        )
        # 随机打乱,批处理,使用多个epoch
        dataset = dataset.shuffle(shuffle_biffer).batch(batch_size).repeat(num_epoch)
        # 生成迭代器
        iterator = dataset.make_initializable_iterator()
        image, label = iterator.get_next()
        if features_name:
            return {str(features_name): image}, label

        return image, label

    return fn


if __name__ == '__main__':

    import time
    import matplotlib.pyplot as plt
    import tensorflow as tf
    import imageio
    import numpy as np
    import os
    import data_prepare.picture_preprocess as pre_process

    from functools import partial

    TFRECORD_FLIE = ""

    HEIGHT = 299
    WEIGHT = 299
    CHANNEL = 3

    BATCH_SIZE = 1
    SHUFFLE_BIFFER = 360
    NUM_EPOCH = 1


    # 解析
    def _parse(record, height, width, channel=3):
        features = tf.parse_single_example(
            record,
            features={
                "image": tf.FixedLenFeature([], tf.string),
                "label": tf.FixedLenFeature([], tf.int64)
            }
        )
        decoded_image = tf.image.decode_jpeg(features["image"], channels=channel)
        if decoded_image.dtype != tf.float32:
            decoded_image = tf.image.convert_image_dtype(decoded_image, tf.float32)
        image = tf.image.resize_images(decoded_image, (height, width))
        image.set_shape([height, width, channel])

        return image, features["label"]


    def gen_dataset_input_fn(file_pattern, height, width, channel=3,
                             shuffle_biffer=360, batch_size=36,
                             num_epoch=1, features_name=None,
                             ):
        def fn():
            # 获取tfrecord文件
            files = tf.train.match_filenames_once(file_pattern)
            # 生成数据集
            dataset = tf.data.TFRecordDataset(files)
            # 解析tfrecord,将二进制转换为tensor
            dataset = dataset.map(lambda record: (_parse(record, height, width, channel)))
            # 图片预处理
            dataset = dataset.map(
                lambda image, label: (pre_process.preprocess_for_train(image, height, width, None), label)
            )
            # 随机打乱,批处理,使用多个epoch
            dataset = dataset.shuffle(shuffle_biffer).batch(batch_size).repeat(num_epoch)
            # 生成迭代器
            iterator = dataset.make_initializable_iterator()
            image, label = iterator.get_next()
            if features_name:
                return {str(features_name): image}, label

            return image, label

        return fn


    if __name__ == '__main__':
        file_pattern = "../dataset/flower_tfrecord/train_data.*"
        files = tf.train.match_filenames_once(file_pattern)
        # 生成数据集
        dataset = tf.data.TFRecordDataset(files)
        # 解析tfrecord,将二进制转换为tensor
        dataset = dataset.map(lambda record: (_parse(record, 299,299,3)))
        # 图片预处理
        dataset = dataset.map(
            lambda image, label: (pre_process.preprocess_for_train(image, 299, 299, None), label)
        )
        # 随机打乱,批处理,使用多个epoch
        dataset = dataset.shuffle(360).batch(3).repeat(1)
        # 生成迭代器
        iterator = dataset.make_initializable_iterator()
        image, label = iterator.get_next()

        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            tf.local_variables_initializer().run()
            sess.run(iterator.initializer)
            while True:
                print(sess.run(image).shape)


        #
