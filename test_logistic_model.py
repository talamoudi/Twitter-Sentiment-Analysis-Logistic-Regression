from utils import *
from itertools import chain



if __name__ == "__main__":
    W, frequencies = load_model("./Results/twitter_sentiment.model")
    Xtest = read_data("./Data/testing_data.csv")
    Ytest = read_data("./Data/testing_labels.csv")
    Ytest = list( chain.from_iterable(Ytest) )
    Xtest, Ytest = extract_features(Xtest, Ytest, frequencies)
    test_logistic_model(Xtest, Ytest, W)