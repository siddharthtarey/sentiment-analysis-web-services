from sklearn.model_selection import train_test_split
from sklearn_deltatfidf import DeltaTfidfVectorizer
import tensorflow as tf
import numpy as np
import math
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression

from project.Preprocess import preProcess

# this function calculates the delta-tf,idf for a given corpus of documents
def deltaMatrix(stemmedList,score):
    v = DeltaTfidfVectorizer()
    v.ngram_range = (2, 2)
    result = v.fit_transform(stemmedList, score)

    return result


# this function creates the ANN model, trains the model and tests it
def createModel(dMatrix,labels,epochs):
    # split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(dMatrix, labels, test_size=0.33, random_state=42)

    # number of hidden nodes in the first hidden layer
    nodesHl1 = 900

    nodesHl2 = 250
    nodesHl3 = 50

    # exptract values, indices and shape of the sparse matrix
    coo_train = X_train.tocoo()
    indices = np.mat([coo_train.row, coo_train.col]).transpose()
    values = coo_train.data
    shape = coo_train.shape

    # tensor flow placeholders for the input and output
    x = tf.sparse_placeholder(tf.float64)
    y = tf.placeholder('float')

    # one hot labeling for training and testing
    yLabel = tf.one_hot(y_train, 2, axis=-1)
    yTestLabel = tf.one_hot(y_test, 2, axis=-1)

    # initiate the ANN model
    prediction = createNN(x,X_train.shape[1],nodesHl1,nodesHl2,nodesHl3)
    # cost function defined
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction,labels=y))
    # optimization
    optimized = tf.train.AdamOptimizer().minimize(cost)


    # intiate a tensor flow session
    with tf.Session() as sess:
        # initialize the tf variables
        sess.run(tf.global_variables_initializer())

        yLabel = yLabel.eval()
        yTestLabel = yTestLabel.eval()

        epochLoss = 0
        # begin training of the neural net
        for epoch in range(epochs):
            epochLoss = 0
            # iterate the training data throuh the model
            i,c = sess.run([optimized,cost], feed_dict={x:tf.SparseTensorValue(indices, values, shape),y:yLabel})
            epochLoss += c

        # extract data, shape and indices of the test data
        coo_test = X_test.tocoo()
        indices_test = np.mat([coo_test.row, coo_test.col]).transpose()
        values_test = coo_test.data
        shape_test = coo_test.shape

        # calculate the accuracy of the data
        correct_prediction = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

        #print the accuracy
        print("For ",epochs, " epochs")

        print("Test Accuracy: ",(sess.run(accuracy,feed_dict={x:tf.SparseTensorValue(indices_test, values_test, shape_test),y:yTestLabel})))




# this function creates hidden and output layers for the ANN
def createNN(x,inputNodes,nodesHl1,nodesHl2,nodesHl3):

    # create 1 hidden layer
    hiddenLayer1 = {'weights': tf.Variable(tf.random_normal([inputNodes,nodesHl1],dtype=tf.float64)),
                    'biases': tf.Variable(tf.random_normal([nodesHl1],dtype=tf.float64))}

    # hiddenLayer2 = {'weights': tf.Variable(tf.random_normal([nodesHl1,nodesHl2],dtype=tf.float64)),
    #                 'biases': tf.Variable(tf.random_normal([nodesHl2],dtype=tf.float64))}
    #
    # hiddenLayer3 = {'weights': tf.Variable(tf.random_normal([nodesHl2, nodesHl3],dtype=tf.float64)),
    #                 'biases': tf.Variable(tf.random_normal([nodesHl3],dtype=tf.float64))}

    # create an output layer of 2 nodes
    output = {'weights': tf.Variable(tf.random_normal([nodesHl1, 2],dtype=tf.float64)),
              'biases': tf.Variable(tf.random_normal([2],dtype=tf.float64))}

    # perform multiplication of input layer and hidden ayer
    # then perform addition of biases
    l1 = tf.add(tf.sparse_tensor_dense_matmul(x,hiddenLayer1['weights']),hiddenLayer1['biases'])
    l1 = tf.nn.leaky_relu(l1)

    # l2 = tf.add(tf.matmul(l1, hiddenLayer2['weights']), hiddenLayer2['biases'])
    # l2 = tf.nn.leaky_relu(l2)
    #
    # l3 = tf.add(tf.matmul(l2, hiddenLayer3['weights']), hiddenLayer3['biases'])
    # l3 = tf.nn.leaky_relu(l3)

    l4 = tf.add(tf.matmul(l1, output['weights']), output['biases'])

    return l4



def main():
    p = preProcess()

    # fetched the cleaned data
    stemmedList, score = p.main()

    # fetch the delta-tf.idf matrix
    tfidfMatrix = deltaMatrix(stemmedList,score)

    # create model for ANN and print accuracy
    for epochs in range(100,1001,100):

        createModel(tfidfMatrix,score,epochs)
    # test Delta tf.idf against SVM linear kernel
    denseMatrix = tfidfMatrix.todense()
    clf = svm.SVC(kernel='linear', C=1)
    accuracy = cross_val_score(clf,denseMatrix,score, cv=10, scoring='accuracy')
    print("SVM with linear kernel ",accuracy)

    # test Delta tf.idf against SVM rbf kernel
    clf = svm.SVC(kernel='rbf', C=1)
    accuracy = cross_val_score(clf, denseMatrix, score, cv=10, scoring='accuracy')
    print("SVM with rbf kernel ", accuracy)

    # test Delta tf.idf against logistic regression
    logreg = LogisticRegression(penalty='l2',max_iter=500)
    accuracyLogReg = cross_val_score(logreg, tfidfMatrix, score, cv=10, scoring='accuracy')
    print("Logistic Regression accuracy ",accuracyLogReg)




main()

