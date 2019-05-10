import pandas as pd
import numpy as np
import tensorflow as tf
import FeatureImp
import datetime

heartdf = pd.read_csv("processed.cleveland.data", header=None,
                      names=['age', 'sex', 'chest pain type', 'resting blood pressure', 'serum cholestoral in mg/dl',
                             ' fasting blood sugar', 'resting electrocardiographic results',
                             ' maximum heart rate achieved', ' exercise induced angina', 'oldpeak',
                             'the slope of the peak exercise ST segment',
                             'number of major vessels (0-3) colored by flourosopy', 'thal', 'target'])

heartdf.replace('?', np.nan, inplace=True)  # Replace ? values
ndf2 = heartdf.dropna()
ndf2.to_csv("heart-disease-cleaveland.txt", sep=",", index=False, header=None, na_rep=np.nan)

input = ['age', 'sex', 'chest pain type', 'resting blood pressure', 'serum cholestoral in mg/dl',
         ' fasting blood sugar', 'resting electrocardiographic results',
         ' maximum heart rate achieved', ' exercise induced angina', 'oldpeak',
         'the slope of the peak exercise ST segment',
         'number of major vessels (0-3) colored by flourosopy', 'thal', 'target']

ndf2.loc[ndf2['target'] != 0] = 1
nlabels = ndf2['target']

'''
# data scaling       The performance is bad, so I remove this part from my model
for i in input:
    a = ''.join(i)
    ndf2[a] = (ndf2[a].astype('float') - ndf2[a].astype('float').min()) / ndf2[a].astype('float').max() - ndf2[
        a].astype('float').min()
    # print(ndf2[a])
'''

nfeatures = ndf2[FeatureImp.feature_weight()].values

nfeatures, nlabels = np.array(nfeatures), np.array(nlabels)
# print(len(features), len(labels))

split_frac = 0.6
n_columns = len(nfeatures)
split_propotion = int(split_frac * n_columns)

train_X, train_Y = nfeatures[:split_propotion], nlabels[:split_propotion]

test_X, test_Y = nfeatures[split_propotion:], nlabels[split_propotion:]

n_labels = 2
n_features = 8

lr = 0.001
n_epochs = 2000
hidden_layer_size = 3
keep_prob = 0.5
checkpoints_dir = "./checkpoints"
logdir = "tensorboard/" + datetime.datetime.now().strftime(
    "%Y%m%d-%H%M%S") + "/"


def train():
    tf.reset_default_graph()
    inputs = tf.placeholder(tf.float32, [None, 8], name='inputs')
    labels = tf.placeholder(tf.int32, [None, ], name='labels')
    labels_one_hot = tf.one_hot(labels, n_labels)
    w1 = tf.Variable(tf.truncated_normal([n_features, hidden_layer_size], stddev=0.1), name='w1')
    b1 = tf.Variable(tf.zeros([hidden_layer_size]), name='b1')
    h1_out = tf.nn.relu(tf.matmul(inputs, w1) + b1)
    h1_out = tf.nn.dropout(h1_out, rate=keep_prob)
    w2 = tf.Variable(tf.truncated_normal([hidden_layer_size, n_labels], stddev=0.1), name='w1')
    b2 = tf.Variable(tf.zeros([n_labels]), name='b1')
    y = tf.matmul(h1_out, w2) + b2
    logits = tf.nn.sigmoid(y, name='logits')
    entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels_one_hot)
    error = tf.reduce_mean(entropy, name='error')
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels_one_hot, 1), name="pred")
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"), name='accuracy')
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(error)
    init = tf.global_variables_initializer()

    # tensorboard
    tf.summary.histogram('w1', w1)
    tf.summary.histogram('b1', b1)
    tf.summary.histogram('w2', w2)
    tf.summary.histogram('b2', b2)
    tf.summary.scalar("training loss", error)
    tf.summary.scalar("training accuracy", accuracy)
    summary_op = tf.summary.merge_all()
    all_saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        file_writer = tf.summary.FileWriter(logdir, sess.graph)
        total_accuracy = 0
        for epoch in range(n_epochs):
            summary, _, accuracy_value, loss = sess.run([summary_op, optimizer, accuracy, error],
                                                        {inputs: train_X, labels: train_Y})
            # print("Train x=",train_X)
            print("Epoch: {0} ; training loss: {1};Accuracy:{2}".format(epoch, loss, accuracy_value))
            total_accuracy += accuracy_value
            if epoch % 10 == 0:
                save_path = all_saver.save(sess, checkpoints_dir +
                                           "/trained_model.ckpt",
                                           global_step=epoch)
                print("Saved model to %s" % save_path)
            file_writer.add_summary(summary, epoch + 1)
        print("The final training accuracy is:", accuracy_value)
        print("During training, the average training accuracy is:", total_accuracy / n_epochs)


def eval():
    total_acc = 0
    step = 10
    for i in range(step):
        with tf.Session() as sess:
            last_check = tf.train.latest_checkpoint('./checkpoints')
            saver = tf.train.import_meta_graph(last_check + ".meta")
            saver.restore(sess, last_check)
            graph = tf.get_default_graph()
            accuracy = graph.get_tensor_by_name('accuracy:0')
            inputs = graph.get_tensor_by_name('inputs:0')
            labels = graph.get_tensor_by_name('labels:0')
            accuracyV = sess.run([accuracy], {inputs: test_X, labels: test_Y})
            for j in accuracyV:
                print('At evaluate step', i, 'the accuracy is: ', j)
                total_acc += j

    print("After training, the average cross validation evaluation accuracy is:", total_acc / step)


def testInput():
    df = pd.read_csv('test_dataset.csv', header=None,
                     names=['age', 'sex', 'chest pain type', 'resting blood pressure', 'serum cholestoral in mg/dl',
                            ' fasting blood sugar', 'resting electrocardiographic results',
                            ' maximum heart rate achieved', ' exercise induced angina', 'oldpeak',
                            'the slope of the peak exercise ST segment',
                            'number of major vessels (0-3) colored by flourosopy', 'thal', 'target'])
    df.loc[df['target'] != 0] = 1
    inputy = df['target']
    inputx = df[FeatureImp.feature_weight()].values
    inputx = np.array(inputx)
    inputy = np.array(inputy)
    with tf.Session() as sess:
        last_check = tf.train.latest_checkpoint('./checkpoints')
        saver = tf.train.import_meta_graph(last_check + ".meta")
        saver.restore(sess, last_check)
        graph = tf.get_default_graph()
        accuracy = graph.get_tensor_by_name('accuracy:0')
        inputs = graph.get_tensor_by_name('inputs:0')
        labels = graph.get_tensor_by_name('labels:0')
        accuracyV = sess.run([accuracy], {inputs: inputx, labels: inputy})
    print(accuracyV)
    return accuracyV


def test():
    total_acc = 0
    step = 10
    for i in range(step):
        with tf.Session() as sess:
            last_check = tf.train.latest_checkpoint('./checkpoints')
            saver = tf.train.import_meta_graph(last_check + ".meta")
            saver.restore(sess, last_check)
            graph = tf.get_default_graph()
            accuracy = graph.get_tensor_by_name('accuracy:0')
            inputs = graph.get_tensor_by_name('inputs:0')
            labels = graph.get_tensor_by_name('labels:0')
            accuracyV = sess.run([accuracy], {inputs: nfeatures, labels: nlabels})
            for j in accuracyV:

                print('At test step', i, 'the accuracy is: ', j)
                total_acc += j
    print("During testing, the average testing accuracy is:", total_acc / step)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "eval", "testInput", "test"])

    args = parser.parse_args()

    if (args.mode == "train"):
        print("Training Run")
        train()
    elif (args.mode == "eval"):
        print("Evaluation run")
        eval()
    elif (args.mode == "testInput"):
        print("Testing input run")
        testInput()
    elif (args.mode == "test"):
        print("Testing run")
        test()
