# Imports
import datetime
import pickle

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from evaluation import *


def load_data(sequenceFile, labelFile):
    """
    加载数据，将本地持久化的pkl文件加载并划分成训练集，验证集以及测试集
    :param sequenceFile: 输入特征的本地持久化pkl文件
    :param labelFile: 真实标签的本地持久化pkl文件
    :param bucketing: 是否进行bucketing操作。bucketing操作可以在训练时减少mini-batch的padding算法复杂度
    :return:train_set: 训练集元组（输入，标签）
            valid_set: 验证集元组（输入，标签）
            test_set: 测试集元组（输入，标签）
    """
    sequences = pickle.load(open(sequenceFile, 'rb'))  # 读取输入序列文件
    labels = pickle.load(open(labelFile, 'rb'))  # 读取真实标签文件
    labels = np.squeeze(np.reshape(labels, (-1, 1)))
    seq_len = np.array([len(seq) for seq in sequences])  # 获取每个样本的长度

    # 下面从正样本和负样本中分别随机抽取相同比例的样本，并分为训练集、测试集、验证集
    dataSize = len(labels)  # 获取数据集的样本总数
    ind_p = np.squeeze(np.argwhere(labels == 1))  # 找到正例样本的下标
    ind_f = np.squeeze(np.setdiff1d(np.arange(dataSize), ind_p))  # 负样本的下标

    per_ind_p = np.random.permutation(ind_p)  # 生成正样本随机排列数，作为数据集的随机索引
    per_ind_f = np.random.permutation(ind_f)  # 生成负样本随机排列数，作为数据集的随机索引

    ind1 = int(0.1 * len(ind_p))
    ind2 = int(0.1 * len(ind_f))
    testP_ind = ind_p[:ind1]
    valP_ind = ind_p[ind1:ind1 * 2]
    trainP_ind = ind_p[ind1 * 2:]
    testf_ind = ind_f[:ind2]
    valf_ind = ind_f[ind2:ind2 * 2]
    trainf_ind = ind_f[ind2 * 2:]
    test_indices = np.random.permutation(np.concatenate((testP_ind, testf_ind)))  # 测试集的样本下标
    valid_indices = np.random.permutation(np.concatenate((valP_ind, valf_ind)))  # 训练集的样本下标
    train_indices = np.random.permutation(np.concatenate((trainP_ind, trainf_ind)))  # 训练集的样本下标
    """
    #ind = np.random.permutation(dataSize)  # 生成随机排列数，作为数据集的随机索引
    nTest = int(0.10 * dataSize)  # 划分数据集的10%作为验证集
    nValid = int(0.10 * dataSize)  # 划分数据集的10%作为测试集

    test_indices = ind[:nTest]  # 获取测试集的索引
    valid_indices = ind[nTest:nTest + nValid]  # 获取验证集的索引
    train_indices = ind[nTest + nValid:]  # 获取训练集的索引
    """
    train_set_x = sequences[train_indices]  # 根据训练集索引数组得到训练集的特征输入列表
    train_set_y = labels[train_indices]  # 根据训练集索引数组得到训练集的真实标签列表
    test_set_x = sequences[test_indices]  # 根据测试集索引数组得到测试集的特征输入列表
    test_set_y = labels[test_indices]  # 根据测试集索引数组得到测试集的真实标签列表
    valid_set_x = sequences[valid_indices]  # 根据验证集索引数组得到验证集的特征输入列表
    valid_set_y = labels[valid_indices]  # 根据验证集索引数组得到验证集的真实标签列表

    return train_set_x, train_set_y, valid_set_x, valid_set_y, test_set_x, test_set_y


def one_hot(labels, n_class=2):
    """ One-hot 编码 """
    expansion = np.eye(n_class)
    y = expansion[:, labels].T
    assert y.shape[1] == n_class, "Wrong number of labels!"

    return y


def get_batches(X, y, batch_size=100):
    """ Return a generator for batches """
    n_batches = len(X) // batch_size
    X, y = X[:n_batches * batch_size], y[:n_batches * batch_size]
    # Loop over batches and yield
    for b in range(0, len(X), batch_size):
        yield X[b:b + batch_size], y[b:b + batch_size]


def model(dataFile, labelFile, lstm_size, lstm_layers, batch_size, seq_len, gap_len, learning_rate, epochs, keep_prob):
    X_tr, lab_tr, X_vld, lab_vld, X_test, lab_test = load_data(dataFile, labelFile)
    X_tr = X_tr[:, -(seq_len + gap_len):-gap_len, :]
    X_vld = X_vld[:, -(seq_len + gap_len):-gap_len, :]
    X_test = X_test[:, -(seq_len + gap_len):-gap_len, :]
    y_tr = one_hot(np.squeeze(lab_tr))
    y_vld = one_hot(np.squeeze(lab_vld))
    y_test = one_hot(np.squeeze(lab_test))

    # Fixed
    n_classes = 2
    n_channels = 51

    # train_set, valid_set, test_set = load_data(dataFile, labelFile, bucketing=True)

    graph = tf.Graph()

    # Construct placeholders
    with graph.as_default():
        inputs_ = tf.placeholder(tf.float32, [None, None, n_channels], name='inputs')
        labels_ = tf.placeholder(tf.float32, [None, n_classes], name='labels')
        keep_prob_ = tf.placeholder(tf.float32, name='keep')
        learning_rate_ = tf.placeholder(tf.float32, name='learning_rate')

        # Convolutional layers
        # filters 是卷积核数量（Integer, the dimensionality of the output space）
        # with graph.as_default():
        # (batch, 128, 9) --> (batch, 128, 10)
        with tf.name_scope('conv1'):
            conv1 = tf.layers.conv1d(inputs=inputs_, filters=100, kernel_size=6, strides=1,
                                     padding='same', activation=tf.nn.relu)
        # n_ch = n_channels * 2      # n_ch是卷积后的特征数量
        n_ch = 100

        with tf.name_scope('LSTM_in'):
            # with graph.as_default():
            # Construct the LSTM inputs and LSTM cells
            lstm_in = tf.transpose(conv1, [1, 0, 2])  # reshape into (seq_len, batch, channels)
            lstm_in = tf.reshape(lstm_in, [-1, n_ch])  # Now (seq_len*N, n_channels)

            # To cells
            """
            tf.layers.dense()
            This layer implements the operation: outputs = activation(inputs.kernel + bias) Where activation is the 
            activation function passed as the activation argument (if not None), kernel is a weights matrix created by 
            the layer, and bias is a bias vector created by the layer (only if use_bias is True).
            此函数参数默认加bias
            kernel_initializer: Initializer function for the weight matrix. If None (default), 
            weights are initialized using the default initializer used by tf.get_variable.
            """
            lstm_in = tf.layers.dense(lstm_in, lstm_size, activation=None)  # or tf.nn.relu, tf.nn.sigmoid, tf.nn.tanh?

            # Open up the tensor into a list of seq_len pieces
            lstm_in = tf.split(lstm_in, seq_len, 0)

        """
        BasicRNNCell是最基本的RNNcell单元。
        输入参数：num_units：RNN层神经元的个数
        input_size（该参数已被弃用）
        activation: 内部状态之间的激活函数
        reuse: Python布尔值, 描述是否重用现有作用域中的变量

        # 使用 DropoutWrapper 类来实现 dropout 功能，output_keep_prob 控制输出的 dropout 概率 

        """
        # Add LSTM layers
        with tf.name_scope('RNN'):
            lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
            drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob_)
            cell = tf.contrib.rnn.MultiRNNCell([drop] * lstm_layers)
            initial_state = cell.zero_state(batch_size, tf.float32)

            # with graph.as_default():
            """
            单层rnn: tf.contrib.rnn.static_rnn：
            参数：inputs是长为T的列表A length T list of inputs, each a `Tensor` of shape
                        [batch_size, input_size]`, or a nested tuple of such elements
                  dtype是初始状态和期望输出的数据类型
            输出： A pair (outputs, state) where:
                  - outputs is a length T list of outputs (one for each input), or a nested
                    tuple of such elements.
                  - state is the final state
            还有rnn中加dropout

            """

            outputs, final_state = tf.contrib.rnn.static_rnn(cell, lstm_in, dtype=tf.float32,
                                                             initial_state=initial_state)
        """
        final_outputs = final_state[layer_size - 1][1]  # 返回最后一层最后一个状态元组的第二个张量，作为输出
        preds = tf.matmul(final_outputs, weight['out']) + bias['out']
        probs = tf.sigmoid(preds)
        """
        # We only need the last output tensor to pass into a classifier
        logits = tf.layers.dense(outputs[-1], n_classes, name='logits')

        with tf.name_scope('cross_entropy'):
            # Cost function and optimizer
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels_))
        tf.summary.scalar("loss", cost)
        # optimizer = tf.train.AdamOptimizer(learning_rate_).minimize(cost) # No grad clipping

        with tf.name_scope('train'):
            # Grad clipping
            # tf.train.AdamOptimizer函数默认参数ersilon = 1e-08
            train_op = tf.train.AdamOptimizer(learning_rate_)

            gradients = train_op.compute_gradients(cost)

        """
        tf.clip_by_value:
        Given a tensor t, this operation returns a tensor of the same type and shape as t with its values clipped to 
        clip_value_min and clip_value_max. Any values less than clip_value_min are set to clip_value_min. 
        Any values greater than clip_value_max are set to clip_value_max.
        """
        with tf.name_scope('clip_value'):
            capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients]
            optimizer = train_op.apply_gradients(capped_gradients)
        """
        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                correct_pred = tf.equal(tf.cast(tf.greater(preds, 0.5), tf.float32), tf.cast(labels_, tf.float32))
            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        """
        with tf.name_scope('accuracy'):
            # Accuracy
            y_pred = tf.argmax(logits, 1);
            y_true = tf.argmax(labels_, 1)
            with tf.name_scope('correct_prediction'):
                correct_pred = tf.equal(y_pred, y_true)  # tf.argmax就是返回最大的那个数值所在的下标
            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')  # tf.cast：用于改变某个张量的数据类型
        tf.summary.scalar("accuracy", accuracy)
        """
        
        with tf.name_scope('PR'):
            
            precision, TPFP = tf.metrics.precision_at_thresholds(labels_, logits, name='precision',thresholds=201)
            recall, TPFN = tf.metrics.recall_at_thresholds(labels_, logits, name='recall',thresholds=201)
            TP, TP_update = tf.metrics.true_positives_at_thresholds(labels_, logits, name='TP',thresholds=201)
            TN, TN_updata = tf.metrics.true_negatives_at_thresholds(labels_, logits, name='TN',thresholds=201)
            FP, FP_update = tf.metrics.false_positives_at_thresholds(labels_, logits, name='FP',thresholds=201)
            FN, FN_update = tf.metrics.false_negatives_at_thresholds(labels_, logits, name='FN',thresholds=201)

            summary_lib.pr_curve_raw_data(name='prc', true_positive_counts=TP, false_positive_counts=FP,
                                          true_negative_counts=TN, false_negative_counts=FN, precision=precision,
                                          recall=recall, display_name='PR Curve',num_thresholds=201)
            summary_lib.scalar('f1_max', tf.reduce_max(2.0 * precision * recall / tf.maximum(precision + recall, 1e-7)))
            
            _, update_op = summary_lib.pr_curve_streaming_op('foo',
                                                             predictions=logits,
                                                             labels=labels_,
                                                             num_thresholds=201)
        """
    validation_acc = []
    validation_loss = []

    train_acc = []
    train_loss = []

    with graph.as_default():
        saver = tf.train.Saver()
        merged = tf.summary.merge_all()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config, graph=graph)

    with session as sess:
        train_writer = tf.summary.FileWriter(LOGDIR + hparam_str,
                                             session.graph)
        test_writer = tf.summary.FileWriter('F://301/github/septic-shock/code/test')
        sess.run(tf.global_variables_initializer())
        iteration = 1

        for e in range(epochs):
            # Initialize
            state = sess.run(initial_state)

            # Loop over batches
            for x, y in get_batches(X_tr, y_tr, batch_size):

                # Feed dictionary
                feed = {inputs_: x, labels_: y, keep_prob_: keep_prob,
                        initial_state: state, learning_rate_: learning_rate}

                loss, output, _, state, acc, summary = sess.run([cost, outputs, optimizer, final_state, accuracy,
                                                                 merged], feed_dict=feed)

                train_acc.append(acc)
                train_loss.append(loss)
                train_writer.add_summary(summary, e)

                """
                # Print at each 5 iters
                if (iteration % 5 == 0):
                    print("Epoch: {}/{}".format(e, epochs),
                          "Iteration: {:d}".format(iteration),
                          "Train loss: {:6f}".format(loss),
                          "Train acc: {:.6f}".format(acc))
                """
                # 每经过34次iteration计算交叉验证集的损失函数以及正确率等指标
                # 这里选择34次是因为训练集有34个batch，每经过34次iteration就相当于在训练集上完成一遍训练
                if (iteration % 34 == 0):

                    # Initiate for validation set
                    val_state = sess.run(cell.zero_state(batch_size, tf.float32))

                    val_acc_ = []
                    val_loss_ = []
                    val_pred = []
                    val_true = []
                    for x_v, y_v in get_batches(X_vld, y_vld, batch_size):
                        # Feed
                        feed = {inputs_: x_v, labels_: y_v, keep_prob_: 1.0, initial_state: val_state}

                        # Loss
                        loss_v, state_v, y_pred_v, y_true_v, acc_v, = sess.run(
                            [cost, final_state, y_pred, y_true, accuracy], feed_dict=feed)
                        val_pred.append(y_pred_v)
                        val_true.append(y_true_v)
                        val_acc_.append(acc_v)
                        val_loss_.append(loss_v)
                        # test_writer.add_summary(summary, e)
                    auc_v = auc(val_true, val_pred)
                    precision_v, recall_v = precision_recall(val_true, val_pred)
                    # Print info
                    """
                    print("Validation: Epoch: {}/{}".format(e, epochs),
                          "Iteration: {:d}".format(iteration),
                          "loss: {:6f}".format(np.mean(val_loss_)),
                          "acc: {:.6f}".format(np.mean(val_acc_)),
                          "auc:{:.2f}".format(auc_v),
                          "precision: {:.4f}".format(precision_v),
                          "recall: {:.4f}".format(recall_v))

                    """
                    # Store
                    validation_acc.append(np.mean(val_acc_))
                    validation_loss.append(np.mean(val_loss_))

                # Iterate
                iteration += 1

        saver.save(sess, "checkpoints-crnn/har.ckpt")
        """
        print("length of Output is:{}".format(len(output)),
            "and output 1 is:{}".format(output[-1].shape))
        print("length of state is:{}".format(len(state)),
            "and state 1 is:{}".format(state[0][-1].shape))
        #print("output[-1] == state[1][-1] is: {}".format((output[-1]==state[1][-1])))
        """
    # Plot training and test loss
    t = np.arange(1, iteration)

    plt.figure(figsize=(12, 6))
    plt.title('hparam:' + hparam_str)
    plt.subplot(121)
    plt.plot(t, np.array(train_loss), 'r-', t[t % 34 == 0], np.array(validation_loss), 'b*')
    plt.ylim(-0.1, 1.0)
    plt.xlabel("iteration")
    plt.ylabel("Loss")
    plt.legend(['train', 'validation'], loc='upper right')
    # plt.show()

    # Plot Accuracies
    # plt.figure(figsize=(6, 6))
    plt.subplot(122)

    plt.plot(t, np.array(train_acc), 'r-', t[t % 34 == 0], validation_acc, 'b*')
    plt.ylim(0.6, 1.01, 0.05)
    plt.xlabel("iteration")
    plt.ylabel("Accuray")
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()

    test_acc = []

    with tf.Session(graph=graph) as sess:
        # Restore
        saver.restore(sess, tf.train.latest_checkpoint('checkpoints-crnn'))
        y_true_test = []
        y_pred_test = []
        for x_t, y_t in get_batches(X_test, y_test, batch_size):
            feed = {inputs_: x_t,
                    labels_: y_t,
                    keep_prob_: 1}

            batch_acc, y_true_t, y_pred_t = sess.run([accuracy, y_true, y_pred], feed_dict=feed)
            test_acc.append(batch_acc)
            y_true_test.append(y_true_t)
            y_pred_test.append(y_pred_t)

        roc_title = "ROC of test samples hparam:" + hparam_str
        prc_title = "precision recall curve:" + hparam_str
        plot_roc(y_true_test, y_pred_test, title=roc_title)
        plot_prc(y_true_test, y_pred_test, title=prc_title)
        class_name = ['sepsis', 'sep_shock']
        precision_t, recall_t = precision_recall(y_true_test, y_pred_test)
        plot_confusion_matric(y_true_test, y_pred_test, classes=class_name, title=hparam_str)
        print("Test accuracy: {:.6f}".format(np.mean(test_acc)),
              "Test precision: {:.4f}".format(precision_t),
              "Test recall: {:.4f}".format(recall_t))


if __name__ == '__main__':
    dataFile = '../data/data.pkl'
    labelFile = '../data/label.pkl'

    # r'submit_results/sub{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    LOGDIR = "F://301/github/septic-shock/results/sub{}/".format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))

    lstm_size = 60  # 2 times the amount of channels
    lstm_layers = 1  # Number of layers
    batch_size = 256  # Batch size
    seq_len = 18  # Number of steps
    gap_len = 6
    learning_rate = 0.0001  # Learning rate (default is 0.001)
    epochs = 200
    keep_prob = 0.8
    """
    for lstm_size in (60, 100, 200):
        for lstm_layers in (1, 2):
            for batch_size in (64, 128, 256):
                hparam_str = "lstm_size=%d,lstm_layer=%d,batch_size=%d,keep_prob=%.E" % (
                    lstm_size, lstm_layers, batch_size, keep_prob)
                print('Starting run for %s' % hparam_str)
                model(dataFile=dataFile, labelFile=labelFile, lstm_size=lstm_size, lstm_layers=lstm_layers,
                      batch_size=batch_size, seq_len=seq_len, gap_len=gap_len, learning_rate=learning_rate,
                      epochs=epochs, keep_prob=keep_prob)
    """

    for seq_len in range(2, 12, 2):
        for gap_len in range(2, 12, 2):
            hparam_str = "ob_len=%d,gap_len=%d" % (
                seq_len, gap_len)
            print('Starting run for %s' % hparam_str)
            model(dataFile=dataFile, labelFile=labelFile, lstm_size=lstm_size, lstm_layers=lstm_layers,
                  batch_size=batch_size, seq_len=seq_len, gap_len=gap_len, learning_rate=learning_rate,
                  epochs=epochs, keep_prob=keep_prob)
