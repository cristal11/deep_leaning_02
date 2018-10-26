import numpy as np
import tensorflow as tf
import os
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.python.slim.nets.inception_v3 as inception_v3

from data_prepare import tfreecord_to_dataset_test as build_dataset


# 不需要的参数前缀
from data_prepare.tfreecord_to_dataset_test import gen_dataset_input_fn
from estimator.data_deal.dataset_input import dataset_input_fn

CHECKPOINT_EXCLUDE_SCOPES = "InceptionV3/Logits,InceptionV3/AuxLogits"

# 需要训练的参数的前缀
TRAINBLE_SCOPES = "InceptionV3/Logits,InceptionV3/AuxLogits"

# 分类数
N_CLASSES = 5
# 学习率
LEARNING_RATE = 0.0001

# 模型文件路径
MODEL_FILE = "./trained_model/inception_v3.ckpt"

# 训练步数
TRAIN_STEP = 10000

# 保存
TRAIN_FILE = "./train_save_model/"

# tfrecord文件路径
TFRECORF_FILES = ""

# 获取训练好的模型的参数
def get_tuned_variable():
    """

    :return:
    """
    exclusions = [scope.strip() for scope in CHECKPOINT_EXCLUDE_SCOPES.split(",")]
    variable_to_restore = []
    # slim.get_variables_to_restore()
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variable_to_restore.append(var)
    return variable_to_restore

# 获取所需要训练的变量列表
def get_trainable_variable():
    """

    :return:
    """
    scopes = [scope.strip() for scope in TRAINBLE_SCOPES.split(",")]
    variable_to_train = []
    for scope in scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope)
        variable_to_train.append(variables)
    return variable_to_train

def cnn_model_fn(features,labels,mode,params):
    """

    :param features:
    :param labels:
    :param mode:
    :param params:
    :return:
    """
    logits_name = 'preditions'
    labels = tf.one_hot(labels,params["nb_classes"])
    # 完善神经网络
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits,_ = inception_v3.inception_v3(inputs=features["image"],
                                             num_classes=params.get("num_classes"))

    # 预测
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions= {
            "class":tf.argmax(logits,1),
            "probabilities":tf.nn.softmax(logits,name="softmax_tensor")
        }
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions
        )

    # 获取需要训练的变量
    trainable_variables = get_trainable_variable()
    # 定义损失
    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels,logits=logits,weights=1.0)

    # 训练过程
    if mode == tf.estimator.ModeKeys.TRAIN:
        # 加载已有的存档点参数

        print("restoring base ckpt")
        variables_to_restore = slim.get_variables_to_restore(exclude=logits_name)
        tf.train.init_from_checkpoint(MODEL_FILE,{v.name.split(":"):v for v in variables_to_restore})
        print("restored base ckpt")

        # 训练
        global_step = tf.train.get_or_create_global_step()
        optimizer = tf.train.AdamOptimizer(params.get("learning_rate",0.0001))
        train_op = optimizer.minimize(loss=tf.losses.get_total_loss(),global_step=global_step)

        return tf.estimator.EstimatorSpec(
            mode=mode,
            train_op=train_op,
            loss=tf.losses.get_total_loss()
        )

    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(labels=tf.argmax(labels, 1),
                                        predictions=tf.argmax(input=logits, axis=1),
                                        name='accuracy')
    }
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        eval_metric_ops=eval_metric_ops
    )


def main(_):

    # 参数
    params = {
        "nb_classes":10,
        "learning_rate":0.0001
    }
    # 构造estimator
    classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn,
        params=params,
        model_dir="./model/mnist_model/"
    )

    # 训练
    train_files = "../data/flower_tfrecord/train_data.*"
    train_input_fn =gen_dataset_input_fn(file_pattern=train_files,
                                         height= 299,
                                         width= 299,
                                         channel=3,
                                         shuffle_biffer=360,
                                         batch_size=36,
                                         num_epoch=1,
                                         features_name="image",
                )

    # 验证
    eval_files = "../data/flower_tfrecord/validation_data.*"
    eval_input_fn = gen_dataset_input_fn(file_pattern=eval_files,
                                          height=299,
                                          width=299,
                                          channel=3,
                                          shuffle_biffer=360,
                                          batch_size=36,
                                          num_epoch=1,
                                          features_name="image",
                                          )

    train_spec = tf.estimator.TrainSpec(
        input_fn=train_input_fn,
    )

    eval_spec = tf.estimator.EvalSpec(
        input_fn=eval_input_fn,
        throttle_secs=60,
        start_delay_secs=120,
    )

    tf.estimator.train_and_evaluate(
        classifier,
        train_spec,
        eval_spec
    )

if __name__ == '__main__':
    tf.app.run()