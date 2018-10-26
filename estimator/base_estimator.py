import numpy as np
import tensorflow as tf


def cnn_model_no_top(x,mode,trainable=False):
    """

    :param features:
    :param model:
    :param trainable:
    :return:
    """
    input_layer = tf.reshape(x, [-1, 28, 28, 1])
    conv1 = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu,
                             trainable=trainable)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu,
                             trainable=trainable)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    pool2_flat = tf.reshape(pool2, shape=[-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu, trainable=trainable)
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=(mode == tf.estimator.ModeKeys.TRAIN))
    return dropout

def cnn_model_fn(features,labels,mode,params):
    """
    用于构造estimatorde model_fn
    :param features:
    :param labels:
    :param model:
    :param params:
    :return:
    """
    logits_name = "prediction"
    labels = tf.one_hot(indices=tf.cast(labels,tf.int32),depth=params["nb_classes"])
    model_no_top = cnn_model_no_top(features["image"],labels,trainable=True)
    logits = tf.layers.dense(
        inputs=model_no_top,
        units=params["nb_classes"],
        name=logits_name
    )
    # 预测
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={
                "result":tf.argmax(logits,1),
                "probabilities":tf.nn.softmax(logits,name="softmax_tensor")
            }
        )

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=logits))
    # 训练
    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()
        optimizer = tf.train.AdamOptimizer(params["learning_rate"])
        train_step = optimizer.minimize(loss,global_step=global_step)

        return tf.estimator.EstimatorSpec(mode=mode,loss=loss,train_op=train_step)
    # 评测
    eval_metric_ops = {
        "accuracy":tf.metrics.accuracy(labels=tf.argmax(labels,1),
                                       predictions=tf.argmax(logits,1),
                                       name="accuracy")
    }
    return tf.estimator.EstimatorSpec(
        mode=mode,
        eval_metric_ops=eval_metric_ops,
        loss=loss
    )

def main(_):
    # 数据准备
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images  # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images  # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    # 参数
    params = {
        "nb_classes":10,
        "learning_rate":0.0001
    }
    # 构造estimator
    classisfiter = tf.estimator.Estimator(
        model_fn=cnn_model_fn,
        params=params,
        model_dir="./model/mnist_model/"
    )
    # 准备训练数据
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"image":train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=1,
        shuffle=True,
        queue_capacity=1000,
        num_threads=3
    )
    # 训练
    classisfiter.train(input_fn=train_input_fn,steps=2000)

    # 验证
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"image": eval_data},
        y=eval_labels,
        batch_size=100,
        num_epochs=1,
        shuffle=False,
        queue_capacity=1000,
        num_threads=3
    )
    eval_result = classisfiter.evaluate(input_fn=eval_input_fn)
    print(eval_result)

if __name__ == '__main__':
    tf.app.run()
