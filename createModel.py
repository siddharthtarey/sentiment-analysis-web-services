from sklearn.model_selection import train_test_split
from sklearn_deltatfidf import DeltaTfidfVectorizer
import tensorflow as tf
import numpy as np

from project.Preprocess import preProcess


def deltaMatrix(stemmedList,score):
    v = DeltaTfidfVectorizer()
    v.ngram_range = (2, 2)
    result = v.fit_transform(stemmedList, score)

    return result

def createModel(dMatrix,labels,epochs):
    X_train, X_test, y_train, y_test = train_test_split(dMatrix, labels, test_size=0.33, random_state=42)

    nodesHl1 = 400
    nodesHl2 = 400
    nodesHl3 = 40

    coo_train = X_train.tocoo()
    indices = np.mat([coo_train.row, coo_train.col]).transpose()
    values = coo_train.data
    shape = coo_train.shape

    x = tf.sparse_placeholder(tf.float64)
    y = tf.placeholder('float')
    yLabel = tf.one_hot(y_train, 2, axis=-1)
    yTestLabel = tf.one_hot(y_test, 2, axis=-1)
    prediction = createNN(x,X_train.shape[1],nodesHl1,nodesHl2,nodesHl3)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction,labels=y))
    optimized = tf.train.AdamOptimizer().minimize(cost)



    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        yLabel = yLabel.eval()
        yTestLabel = yTestLabel.eval()


        for epoch in range(epochs):
            epochLoss = 0
            i,c = sess.run([optimized,cost], feed_dict={x:tf.SparseTensorValue(indices, values, shape),y:yLabel})
            epochLoss += c

        coo_test = X_test.tocoo()
        indices_test = np.mat([coo_test.row, coo_test.col]).transpose()
        values_test = coo_test.data
        shape_test = coo_test.shape

        correct_prediction = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        print("For ",epochs, " epochs")
        print("Test Accuracy: ",(sess.run(accuracy,feed_dict={x:tf.SparseTensorValue(indices_test, values_test, shape_test),y:yTestLabel})))





def createNN(x,inputNodes,nodesHl1,nodesHl2,nodesHl3):

    hiddenLayer1 = {'weights': tf.Variable(tf.random_normal([inputNodes,nodesHl1],dtype=tf.float64)),
                    'biases': tf.Variable(tf.random_normal([nodesHl1],dtype=tf.float64))}

    hiddenLayer2 = {'weights': tf.Variable(tf.random_normal([nodesHl1,nodesHl2],dtype=tf.float64)),
                    'biases': tf.Variable(tf.random_normal([nodesHl2],dtype=tf.float64))}

    hiddenLayer3 = {'weights': tf.Variable(tf.random_normal([nodesHl2, nodesHl3],dtype=tf.float64)),
                    'biases': tf.Variable(tf.random_normal([nodesHl3],dtype=tf.float64))}

    output = {'weights': tf.Variable(tf.random_normal([nodesHl3, 2],dtype=tf.float64)),
              'biases': tf.Variable(tf.random_normal([2],dtype=tf.float64))}

    l1 = tf.add(tf.sparse_tensor_dense_matmul(x,hiddenLayer1['weights']),hiddenLayer1['biases'])
    l1 = tf.nn.leaky_relu(l1)

    l2 = tf.add(tf.matmul(l1, hiddenLayer2['weights']), hiddenLayer2['biases'])
    l2 = tf.nn.leaky_relu(l2)

    l3 = tf.add(tf.matmul(l2, hiddenLayer3['weights']), hiddenLayer3['biases'])
    l3 = tf.nn.leaky_relu(l3)

    l4 = tf.add(tf.matmul(l3, output['weights']), output['biases'])

    return l4



def main():
    p = preProcess()

    stemmedList, score = p.main()

    tfidfMatrix = deltaMatrix(stemmedList,score)

    for epochs in range(1000,5001,1000):

        createModel(tfidfMatrix,score,epochs)



main()

