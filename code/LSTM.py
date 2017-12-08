import pickle
import sys
from time import time

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from tensorflow.contrib import rnn
import matplotlib.pyplot as plt


def load_data(sequenceFile, labelFile, bucketing=False):
    """
    加载数据，将本地持久化的pkl文件加载并划分成训练集，验证集以及测试集
    :param sequenceFile: 输入特征的本地持久化pkl文件
    :param labelFile: 真实标签的本地持久化pkl文件
    :param bucketing: 是否进行bucketing操作。bucketing操作可以在训练时减少mini-batch的padding算法复杂度
    :return:train_set: 训练集元组（输入，标签）
            valid_set: 验证集元组（输入，标签）
            test_set: 测试集元组（输入，标签）
    """
    sequences = np.array(pickle.load(open(sequenceFile, 'rb')))  # 读取输入序列文件
    labels = np.array(pickle.load(open(labelFile, 'rb')))  # 读取真实标签文件
    labels = np.reshape(labels, (-1, 1))
    seq_len = np.array([len(seq) for seq in sequences])

    dataSize = len(labels)  # 获取数据集的样本总数
    ind = np.random.permutation(dataSize)  # 生成随机排列数，作为数据集的随机索引
    nTest = int(0.10 * dataSize)  # 划分数据集的10%作为验证集
    nValid = int(0.10 * dataSize)  # 划分数据集的10%作为测试集

    test_indices = ind[:nTest]  # 获取测试集的索引
    valid_indices = ind[nTest:nTest + nValid]  # 获取验证集的索引
    train_indices = ind[nTest + nValid:]  # 获取训练集的索引

    train_set_x = sequences[train_indices]  # 根据训练集索引数组得到训练集的特征输入列表
    train_set_y = labels[train_indices]  # 根据训练集索引数组得到训练集的真实标签列表
    train_set_len = seq_len[train_indices]  # 根据训练集索引数组得到训练集样本的时间序列长度列表
    test_set_x = sequences[test_indices]  # 根据测试集索引数组得到测试集的特征输入列表
    test_set_y = labels[test_indices]  # 根据测试集索引数组得到测试集的真实标签列表
    test_set_len = seq_len[test_indices]  # 根据测试集索引数组得到测试集样本的时间序列长度列表
    valid_set_x = sequences[valid_indices]  # 根据验证集索引数组得到验证集的特征输入列表
    valid_set_y = labels[valid_indices]  # 根据验证集索引数组得到验证集的真实标签列表
    valid_set_len = seq_len[valid_indices]  # 根据验证集索引数组得到验证集样本的时间序列长度列表

    """
    将列表按列表内元素的长度进行排序，默认升序，返回的列表存储原来内层列表的索引
    bucketing操作，使用dynamic rnn则不需要，它会根据每个batch自动计算最好的输出，不过要更定每个example的 sequence length。
    """
    if bucketing == True:
        def len_argsort(seq):
            return sorted(range(len(seq)), key=lambda x: len(seq[x]))

        train_sorted_index = len_argsort(train_set_x)
        train_set_x = [train_set_x[i] for i in train_sorted_index]
        train_set_y = [train_set_y[i] for i in train_sorted_index]

        valid_sorted_index = len_argsort(valid_set_x)
        valid_set_x = [valid_set_x[i] for i in valid_sorted_index]
        valid_set_y = [valid_set_y[i] for i in valid_sorted_index]

        test_sorted_index = len_argsort(test_set_x)
        test_set_x = [test_set_x[i] for i in test_sorted_index]
        test_set_y = [test_set_y[i] for i in test_sorted_index]

    train_set = (train_set_x, train_set_y, train_set_len)
    valid_set = (valid_set_x, valid_set_y, valid_set_len)
    test_set = (test_set_x, test_set_y, test_set_len)

    return train_set, valid_set, test_set


def padMatrix(seqs):
    lengths = [len(s) for s in seqs]
    n_samples = len(seqs)
    maxlen = np.max(lengths)

    x = np.zeros((n_samples, maxlen, input_size)).astype('int32')
    for idx, s in enumerate(seqs):
        x[idx, :lengths[idx]] = s
    return x


if __name__ == '__main__':
    dataFile = '../rawdata/data_pkl/data.pkl'
    labelFile = '../rawdata/data_pkl/label.pkl'
    layer_size = 2
    batch_size = 128
    hidden_size = 100
    input_size = 51
    output_size = 1
    drop_rate = 0.5
    train_iter = 100
    lr = 0.0001
    # L2_penalty = 1e-2

    seed = 19930308
    rs = np.random.RandomState(seed=seed)

    train_set, valid_set, test_set = load_data(dataFile, labelFile, bucketing=True)
    # train_set=tf.constant(train_set)
    # valid_set=tf.constant(valid_set)
    # test_set=tf.constant(test_set)

    # 定义模型的输入和输出
    X = tf.placeholder(tf.float32, [None, None, input_size], name='input')
    Y = tf.placeholder(tf.float32, [None, output_size], name='output')
    X_len = tf.placeholder(tf.float32, [None], name='series_num')

    # 定义模型输入输出的权重和偏置，并初始化
    weight = {'in': tf.Variable(tf.random_normal([input_size, hidden_size])),
              'out': tf.Variable(tf.random_normal([hidden_size, output_size]))}
    bias = {'in': tf.Variable(tf.constant(0.1, shape=[hidden_size, ])),
            'out': tf.Variable(tf.constant(0.1, shape=[output_size, ]))}

    # todo:cell的输入
    x_batch = tf.shape(X)[0]
    # x_time=tf.shape(X)[1]
    X_in = tf.reshape(X, [-1, input_size])

    X_in = tf.matmul(X_in, weight['in']) + bias['in']
    X_in = tf.reshape(X_in, [x_batch, -1, hidden_size])

    # 构造RNN隐藏层LSTM层
    lstm_cell = rnn.BasicLSTMCell(hidden_size)
    lstm_cell = rnn.DropoutWrapper(lstm_cell, output_keep_prob=drop_rate)
    lstm_cell = rnn.MultiRNNCell([lstm_cell for _ in range(layer_size)])

    init_state = lstm_cell.zero_state(batch_size=x_batch, dtype=tf.float32)  # 初始化状态神经元
    state = init_state

    outputs, states = tf.nn.dynamic_rnn(lstm_cell, inputs=X_in, initial_state=state, sequence_length=X_len)
    final_outputs = states[layer_size - 1][1]  # 返回最后一层最后一个状态元组的第二个张量，作为输出
    preds = tf.matmul(final_outputs, weight['out']) + bias['out']
    probs = tf.sigmoid(preds)

    """
    loss_fit = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=preds, labels=Y))
    with tf.variable_scope("", reuse=True):
        loss_reg = L2_penalty * tf.reduce_sum(tf.square(weight['out']))
        for i in range(layer_size):
            loss_reg = L2_penalty + tf.reduce_sum(
                tf.square(tf.get_variable('rnn/cell_' + str(i) + '/basic_lstm_cell/weights')))
    loss = loss_fit + loss_reg
    """
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=preds, labels=Y))
    # tf.summary.scalar('loss', loss)

    train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss=loss)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_pred = tf.equal(tf.cast(tf.greater(preds, 0.5), tf.float32), tf.cast(Y, tf.float32))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    # tf.summary.scalar('accuracy',accuracy)

    sess = tf.Session()

    writer = tf.summary.FileWriter("/train", sess.graph)
    saver = tf.train.Saver()
    ##### 初始化全局变量
    sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

    # merged_summary_op=tf.summary.merge_all()
    print("Graph setup!")

    N_len = len(train_set[1])
    Ntr = np.arange(0, N_len)
    test_freq = N_len // batch_size
    # setup minibatch indices
    starts = np.arange(0, N_len, batch_size)
    ends = np.arange(batch_size, N_len + 1, batch_size)
    if ends[-1] < N_len:
        ends = np.append(ends, N_len)
    num_batches = len(ends)

    start_time = time()
    total_batches = 0
    bestValidAuc = 0
    for i in range(train_iter):
        epoch_start = time()
        perm = rs.permutation(N_len)
        # perm=np.array(list(map(int,perm)))
        batch = 0
        print("Starting epoch " + "{:d}".format(i))
        for s, e in zip(starts, ends):
            batch_start = time()
            inds = perm[s:e]
            train_set0 = []
            train_set1 = []
            train_set2 = []
            train_set0 = [train_set[0][x] for x in inds]
            train_set1 = [train_set[1][x] for x in inds]
            train_set2 = [train_set[2][x] for x in inds]
            tr_probs, tr_acc, loss_, _ = sess.run([probs, accuracy, loss, train_op],
                                                  feed_dict={X: padMatrix(train_set0),
                                                             Y: np.array(train_set1), X_len: train_set2})
            # tr_auc = roc_auc_score(train_set1, tr_probs)

            print("Batch " + "{:d}".format(batch) + "/" + "{:d}".format(num_batches) + \
                  ", took: " + "{:.3f}".format(time() - batch_start) + ", loss: " + "{:.5f}".format(
                loss_) + " Acc: " + "{:.5f}".format(tr_acc))
            batch = batch + 1
            total_batches = total_batches + 1
            sys.stdout.flush()

        """
        if total_batches % test_freq == 0:  # Check val set every so often for early stopping
            # TODO: may also want to check validation performance at additional X hours back
            # from the event time, as well as just checking performance at terminal time
            # on the val set, so you know if it generalizes well further back in time as well
            test_t = time()
            te_probs, te_acc, te_loss = sess.run([probs, accuracy, loss],
                                                 feed_dict={X: padMatrix(valid_set[0]), Y: np.array(valid_set[1]),
                                                            X_len: valid_set[2]})
            te_auc = roc_auc_score(valid_set[1], te_probs)
            te_prc = average_precision_score(valid_set[1], te_probs)
            print("Epoch " + str(i) + ", seen " + str(total_batches) + " total batches. Testing Took " + \
                  "{:.2f}".format(time() - test_t) + \
                  ". OOS, " + str(0) + " hours back: Loss: " + "{:.5f}".format(te_loss) + \
                  " Acc: " + "{:.5f}".format(te_acc) + ", AUC: " + \
                  "{:.5f}".format(te_auc) + ", AUPR: " + "{:.5f}".format(te_prc))
            sys.stdout.flush()
        """
        # summary_str = sess.run(merged_summary_op)
        # writer.add_summary(summary_str,i)

        valid_t = time()
        with tf.name_scope('valid'):
            va_probs, va_acc, va_loss = sess.run([probs, accuracy, loss],
                                                 feed_dict={X: padMatrix(valid_set[0]), Y: np.array(valid_set[1]),
                                                            X_len: valid_set[2]})

            va_auc = roc_auc_score(valid_set[1], va_probs)
            # tf.summary.scalar('auc',va_auc)
            va_prc = average_precision_score(valid_set[1], va_probs)
            # tf.summary.scalar('pr',va_prc)
            print("Epoch " + str(i) + ", seen " + str(total_batches) + " total batches. Testing Took " + \
                  "{:.2f}".format(time() - valid_t) + \
                  ". OOS, " + str(0) + " hours back: Loss: " + "{:.5f}".format(va_loss) + \
                  " Acc: " + "{:.5f}".format(va_acc) + ", AUC: " + \
                  "{:.5f}".format(va_auc) + ", AUPR: " + "{:.5f}".format(va_prc))
            sys.stdout.flush()

        if (va_auc > bestValidAuc):
            bestValidAuc = va_auc
            saver.save(sess, "/Model/model.ckpt", global_step=i)
            best_probs = va_probs
            """            
            test_t=time()
            te_probs, te_acc, te_loss = sess.run([probs, accuracy, loss],
                                                 feed_dict={X: padMatrix(test_set[0]), Y: np.array(test_set[1]),
                                                            X_len: test_set[2]})
            te_auc = roc_auc_score(test_set[1], te_probs)
            te_prc = average_precision_score(test_set[1], te_probs)
            """
            print("测试集运行时间：" + \
                  "{:.2f}".format(time() - valid_t) + \
                  ". OOS, " + str(0) + " hours back: Loss: " + "{:.5f}".format(va_loss) + \
                  " Acc: " + "{:.5f}".format(va_acc) + ", AUC: " + \
                  "{:.5f}".format(va_auc) + ", AUPR: " + "{:.5f}".format(va_prc))
            sys.stdout.flush()
            # bestTestAuc = te_auc

            # bestParams = unzip(tparams)
            print
            '当前测试集最优的AUC:%f' % va_auc

    fpr, tpr, thresholds_roc = roc_curve(valid_set[1], best_probs)
    precision, recall, thresholds_pr = precision_recall_curve(valid_set[1], best_probs)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % bestValidAuc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    plt.savefig('/pic/roc1.png')

    print("Finishing epoch " + "{:d}".format(i) + ", took " + \
          "{:.3f}".format(time() - epoch_start))
