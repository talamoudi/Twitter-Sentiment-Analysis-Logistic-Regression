import numpy as np
import matplotlib.pyplot as plt
from pickle import load, dump




def generate_word_frequencies(tweets, labels):
    assert(len(tweets) == len(labels))
    frequencies = dict()
    for tweet, label in zip(tweets, labels):
        label = int(label)
        for word in tweet:
            if (label, word) in frequencies:
                frequencies[(label, word)] += 1
            else:
                frequencies[(label, word)] = 1

    return frequencies





def extract_features(tweets, labels, frequencies):
    assert(len(tweets) == len(labels))
    data_set = np.array([[1,0,0]]*len(tweets))

    for i in range(len(tweets)):
        for word in tweets[i]:
            if (1, word) in frequencies:
                data_set[i, 1] += frequencies[(1, word)]
            if (0, word) in frequencies:
                data_set[i, 2] += frequencies[(0, word)]

    labels = np.array([float(label) for label in labels])
    return data_set, labels.reshape( (labels.shape[0], 1) )





def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))





def loss(X, Y, Yhat):
    return -1.0 / X.shape[0] * (np.dot(Y.T, np.log(Yhat)) + np.dot((1 - Y).T, np.log(1 - Yhat))).item()


            


def predict_class(X, W):
    return sigmoid( np.dot(X, W) ) >= 0.5





def train_logistic_model(X, Y, lr = 1e-8, epochs = 1000, verbose = False, convergance_duration = 30):
    W = np.zeros( shape = (X.shape[1], 1) )
    Losses, Accuracies = list(), list()
    Losses.append( loss(X, Y, sigmoid( np.dot(X, W) )) )
    Accuracies.append( 1.0 / X.shape[0] * np.sum( Y == predict_class(X, W)) )
    prev_accuracy = 0
    count = 0
    for epoch in range(epochs):
        Yhat = sigmoid( np.dot(X, W) )
        W = W - lr / X.shape[0] * np.dot(X.T, (Yhat - Y))
        Losses.append( loss(X, Y, Yhat) )
        Accuracies.append( 1.0 / X.shape[0] * np.sum(Y == predict_class(X, W)) )
        if verbose:
            print("Iteration {}/{}: Loss = {}, Accuracy = {}".format(epoch + 1, epochs, Losses[-1], Accuracies[-1]) )
        if prev_accuracy == Accuracies[-1] and count < convergance_duration:
            count += 1
        else:
            count = 0
        prev_accuracy = Accuracies[-1]
        if count == convergance_duration:
            break
    return W, Losses, Accuracies





def plot_training(Losses, Accuracies, filename="./Results/loss_and_accuracy.png"):
    plt.subplot(211)
    plt.title('Logistic Regression Loss and Accuracy')
    plt.plot(range(len(Losses)), Losses, c='r')
    plt.ylabel('Loss')
    plt.subplot(212)
    plt.plot(range(len(Accuracies)), Accuracies, c='b')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.savefig(filename)





def test_logistic_model(Xtest, Ytest, W, verbose = True):
    Yhat = predict_class(Xtest, W)
    accuracy = 1.0 / Xtest.shape[0] * np.sum(Ytest == Yhat)
    if verbose:
        print("The accuracy of the model on the test set is : {}".format(accuracy))
    return accuracy





def read_data(filename):
    Data = list()
    infile = open(filename, "r")
    for line in infile:
        line = line[:-1]
        tokens = line.split(",")
        if len(tokens) == 0:
            continue
        Data.append(tokens)
    infile.close()
    return Data





def write_file(filename, data):
    file = open(filename,"w")
    for item in data:
        if type(item) == list:
            for i in range(len(item)):
                if item[i] == '\n':
                    continue
                file.write(item[i])
                if i == len(item) - 1:
                    file.write("\n")
                else:
                    file.write(",")
        else:
            file.write(str(item) + "\n")
    file.close()





def save_model(filename, model):
    file = open(filename, 'wb')
    dump(model, file)
    file.close()





def load_model(filename):
    file = open(filename, 'rb')
    model = load(file)
    file.close()
    return model


