from utils import *
from itertools import chain




if __name__ == "__main__":
    Xtrain = read_data("./Data/training_data.csv")
    Ytrain = read_data("./Data/training_labels.csv")
    Ytrain = list(chain.from_iterable(Ytrain))

    Xtest = read_data("./Data/testing_data.csv")
    Ytest = read_data("./Data/testing_labels.csv")
    Ytest = list(chain.from_iterable(Ytest))

    frequencies = generate_word_frequencies(Xtrain, Ytrain)
    X, Y = extract_features(Xtrain, Ytrain, frequencies)
    Xtest, Ytest = extract_features(Xtest, Ytest, frequencies)

    W, Losses, Accuracies = train_logistic_model(X, Y, lr = 1e-8, epochs = 1300, verbose = True)
    plot_training(Losses, Accuracies)
    
    test_logistic_model(Xtest, Ytest, W)

    save_model("./Results/twitter_sentiment.model", model = (W, frequencies))
