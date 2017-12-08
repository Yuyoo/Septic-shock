# Imports
import tensorflow as tf
import pickle
import numpy as np
import matplotlib.pyplot as plt


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


if __name__ == '__main__':

    dataFile = '../rawdata/data_pkl/data.pkl'
    labelFile = '../rawdata/data_pkl/label.pkl'
    X_tr, lab_tr, X_vld, lab_vld, X_test, labels_test = load_data(dataFile, labelFile)

    y_tr = one_hot(np.squeeze(lab_tr))
    y_vld = one_hot(np.squeeze(lab_vld))
    y_test = one_hot(np.squeeze(labels_test))

    lstm_size = 80  # 2 times the amount of channels
    lstm_layers = 2  # Number of layers
    batch_size = 200  # Batch size
    seq_len = 48  # Number of steps
    learning_rate = 0.0001  # Learning rate (default is 0.001)
    epochs = 200
    keep_prob = 0.6
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
    with graph.as_default():
        # (batch, 128, 9) --> (batch, 128, 10)
        conv1 = tf.layers.conv1d(inputs=inputs_, filters=100, kernel_size=4, strides=1,
                                 padding='same', activation=tf.nn.relu)
        # n_ch = n_channels * 2      # n_ch是卷积后的特征数量
        n_ch = 100

    with graph.as_default():
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
        lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
        drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob_)
        cell = tf.contrib.rnn.MultiRNNCell([drop] * lstm_layers)
        initial_state = cell.zero_state(batch_size, tf.float32)

    with graph.as_default():
        """
        单层rnn: tf.contrib.rnn.static_rnn：
        输入：[步长,batch,input] 
        输出：[n_steps,batch,n_hidden] 
        还有rnn中加dropout

        多层rnn： tf.nn.dynamic_rnn：
        输入：[batch,步长,input] 
        输出：[batch,n_steps,n_hidden] 
        所以我们需要tf.transpose(outputs, [1, 0, 2])，这样就可以取到最后一步的output
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

        # Cost function and optimizer
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels_))
        # optimizer = tf.train.AdamOptimizer(learning_rate_).minimize(cost) # No grad clipping

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

        capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients]
        optimizer = train_op.apply_gradients(capped_gradients)

        # Accuracy
        correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels_, 1))  # tf.argmax就是返回最大的那个数值所在的下标
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')  # tf.cast：用于改变某个张量的数据类型

    validation_acc = []
    validation_loss = []

    train_acc = []
    train_loss = []

    with graph.as_default():
        saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config, graph=graph)

    with session as sess:
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

                loss, _, state, acc = sess.run([cost, optimizer, final_state, accuracy],
                                               feed_dict=feed)
                train_acc.append(acc)
                train_loss.append(loss)

                # Print at each 5 iters
                if (iteration % 5 == 0):
                    print("Epoch: {}/{}".format(e, epochs),
                          "Iteration: {:d}".format(iteration),
                          "Train loss: {:6f}".format(loss),
                          "Train acc: {:.6f}".format(acc))

                # Compute validation loss at every 25 iterations
                if (iteration % 25 == 0):

                    # Initiate for validation set
                    val_state = sess.run(cell.zero_state(batch_size, tf.float32))

                    val_acc_ = []
                    val_loss_ = []
                    for x_v, y_v in get_batches(X_vld, y_vld, batch_size):
                        # Feed
                        feed = {inputs_: x_v, labels_: y_v, keep_prob_: 1.0, initial_state: val_state}

                        # Loss
                        loss_v, state_v, acc_v = sess.run([cost, final_state, accuracy], feed_dict=feed)

                        val_acc_.append(acc_v)
                        val_loss_.append(loss_v)

                    # Print info
                    print("Epoch: {}/{}".format(e, epochs),
                          "Iteration: {:d}".format(iteration),
                          "Validation loss: {:6f}".format(np.mean(val_loss_)),
                          "Validation acc: {:.6f}".format(np.mean(val_acc_)))

                    # Store
                    validation_acc.append(np.mean(val_acc_))
                    validation_loss.append(np.mean(val_loss_))

                # Iterate
                iteration += 1

        saver.save(sess, "checkpoints-crnn/har.ckpt")

    # Plot training and test loss
    t = np.arange(iteration - 1)

    plt.figure(figsize=(6, 6))
    plt.plot(t, np.array(train_loss), 'r-', t[t % 25 == 0], np.array(validation_loss), 'b*')
    plt.xlabel("iteration")
    plt.ylabel("Loss")
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()

    # Plot Accuracies
    plt.figure(figsize=(6, 6))

    plt.plot(t, np.array(train_acc), 'r-', t[t % 25 == 0], validation_acc, 'b*')
    plt.xlabel("iteration")
    plt.ylabel("Accuray")
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()

    test_acc = []

    with tf.Session(graph=graph) as sess:
        # Restore
        saver.restore(sess, tf.train.latest_checkpoint('checkpoints-crnn'))

        for x_t, y_t in get_batches(X_test, y_test, batch_size):
            feed = {inputs_: x_t,
                    labels_: y_t,
                    keep_prob_: 1}

            batch_acc = sess.run(accuracy, feed_dict=feed)
            test_acc.append(batch_acc)
        print("Test accuracy: {:.6f}".format(np.mean(test_acc)))
