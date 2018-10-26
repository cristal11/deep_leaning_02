import numpy as np
import tensorflow as tf
import os
import tensorflow.contrib.slim as slim

def cnn_model_no_top(x, trainable):
    """
    :param features: 原始输入
    :param mode: estimator模式
    :param trainable: 该层的变量是否可训练
    :return: 不含最上层全连接层的模型
    """
    input_layer = tf.reshape(x, [-1, 28, 28, 1])
    conv1 = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu, trainable=trainable)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu, trainable=trainable)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    pool2_flat = tf.reshape(pool2, shape=[-1, 7 * 7 * 64])
    return pool2_flat

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
    model_no_top = cnn_model_no_top(features["image"],
                                    trainable=not (params.get("transfer") or params.get("finetune")))
    with tf.name_scope("finetune"):
        # 允许第二次迁移学习微调
        dense = tf.layers.dense(
            model_no_top,params["1024"],
            activation=tf.nn.relu,
            trainable=params.get("finetune")
        )
    dropout = tf.layers.dropout(inputs=dense,rate=0.4,
                                training= (mode==tf.estimator.ModeKeys.TRAIN))
    logits = tf.layers.dense(dropout,params["nb_classes"],name=logits_name)

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

    # 定义损失
    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels,logits=logits,weights=1.0)

    # 训练过程
    if mode == tf.estimator.ModeKeys.TRAIN:
        # 加载已有的存档点参数
        if params.get("checkpoints") and isinstance(params.get("checkpoints"),(tuple,list)):
            for ckpt in params.get("checkpoint"):
                # [0]是存档点路径,[1]为是否加载倒数第二个全连接参数
                if ckpt[1]:
                    print("restoring base ckpt")
                    variables_to_restore = slim.get_variables_to_restore(exclude=logits_name)
                    tf.train.init_from_checkpoint(ckpt[0],{v.name.split(":"):v for v in variables_to_restore})
                    print("restored base ckpt")
                else:
                    print("restoring transferred ckpt")
                    variables_to_restore = slim.get_variables_to_restore(exclude=[logits_name,"finetune"])
                    tf.train.init_from_checkpoint(ckpt[0], {v.name.split(":"): v for v in variables_to_restore})
                    print("restored transferred ckpt")
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

    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    params = {
        "nb_classes":10,
        "learning_rate":0.0001
    }

    classifer = tf.estimator.Estimator(
        model_fn=cnn_model_fn,
        params=params
    )
    # 开始第一次训练
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x = {"image":train_data},
        y = train_labels,
        batch_size=50,
        shuffle=True,
        num_epochs=3,
        num_threads=3
    )
    classifer.train(input_fn=train_input_fn)


