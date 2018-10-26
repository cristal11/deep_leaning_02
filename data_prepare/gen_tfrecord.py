import numpy
import imageio
import tensorflow as tf
import os
import numpy as np

OUT_DIR = "dataset/flowers_tfrecord/"

VALIDATION_DATA_PERCENTAGE = 10

TEST_DATA_PERCENTAGE = 20

def transform__to_tfrecord(image,label,writer):
    # image_batch = read_picture_files(file_names)

    # writer = tf.python_io.TFRecordWriter(out_file_name)

    example = tf.train.Example(features=tf.train.Features(feature={
        "image":tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
        "label":tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
    }))
    writer.write(example.SerializeToString())

def read_picture(filename):
    img_data = open(filename,"rb").read()
    return img_data

def gen_tfrecord_file_02():
    sub_dirs = [x[0] for x in os.walk("../data/flowers/")]
    with tf.Session() as sess:
        i = 0  # label

        out_file_train_name = os.path.join(OUT_DIR,"train.tfrecords")
        out_file_test_name = os.path.join(OUT_DIR,"test_data.tf.records")
        out_file_validation_name = os.path.join(OUT_DIR,"validation_data.tf.records")
        # 将数据分为训练集,验证集和测试集
        writer_validation = tf.python_io.TFRecordWriter(out_file_validation_name)
        writer_test = tf.python_io.TFRecordWriter(out_file_test_name)
        writer_train = tf.python_io.TFRecordWriter(out_file_train_name)

        for sub_dir in sub_dirs[1:]:
            files = os.listdir(sub_dir)
            file_list = [os.path.join(sub_dir, file) for file in files]
            # image_batch = read_picture_files(file_list)
            j = 0
            for filename in file_list:
                image = read_picture(filename)

                chance = numpy.random.randint(100)

                if chance < VALIDATION_DATA_PERCENTAGE:
                    transform__to_tfrecord(image, i,writer_validation)
                elif chance <(VALIDATION_DATA_PERCENTAGE+TEST_DATA_PERCENTAGE):
                    transform__to_tfrecord(image, i,writer_test)
                else:
                    transform__to_tfrecord(image,i,writer_train)
                print("laoding ---------{}-of-{}-of-{}".format(j,len(file_list),i))
                j += 1
            i += 1
        writer_validation.close()
        writer_train.close()
        writer_test.close()
